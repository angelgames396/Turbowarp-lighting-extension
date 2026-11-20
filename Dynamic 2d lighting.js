// Name: 2D Dynamic Lighting & Shadows
// ID: dynamicLighting2D
// Description: Real-time 2D dynamic lighting, raycast shadows, sprite-silhouette shadows, and Camera v2 compatibility for TurboWarp.
// License: MIT

(function (Scratch) {
  "use strict";

  if (!Scratch.extensions.unsandboxed) {
    throw new Error("2D Dynamic Lighting extension must run unsandboxed!");
  }

  const vm = Scratch.vm;
  const runtime = vm.runtime;
  const renderer = vm.renderer;

  // Pre-allocated typed arrays for Polar Angular Occlusion Binning (zero per-frame GC allocation)
  const NUM_BINS = 720; // 0.5° resolution around 360° for razor-sharp spotlight & point shadows
  const TWO_PI = Math.PI * 2;
  const INV_TWO_PI = 1 / TWO_PI;
  const BIN_STEP_RAD = TWO_PI / NUM_BINS;

  const BIN_OCCUPIED = new Uint8Array(NUM_BINS);
  const BIN_MIN_DIST = new Float32Array(NUM_BINS);
  const COS_TABLE = new Float32Array(NUM_BINS + 1);
  const SIN_TABLE = new Float32Array(NUM_BINS + 1);
  const COS_MID_TABLE = new Float32Array(NUM_BINS);
  const SIN_MID_TABLE = new Float32Array(NUM_BINS);

  for (let i = 0; i <= NUM_BINS; i++) {
    const a = i * BIN_STEP_RAD;
    COS_TABLE[i] = Math.cos(a);
    SIN_TABLE[i] = Math.sin(a);
  }
  for (let i = 0; i < NUM_BINS; i++) {
    const a = (i + 0.5) * BIN_STEP_RAD;
    COS_MID_TABLE[i] = Math.cos(a);
    SIN_MID_TABLE[i] = Math.sin(a);
  }

  class DynamicLightingExtension {
    constructor() {
      // Lighting State
      this.enabled = false;
      this.ambientDarkness = 85; // 0 to 100%
      this.ambientColor = "#000011";
      this.renderMode = "smooth"; // 'smooth' or 'pixelated'
      this.pixelScale = 0.25;

      // Collections
      this.lights = new Map(); // name -> LightObject
      this.obstacles = new Map(); // targetId -> { target, shape: 'bounding box' | 'circle' | 'sprite shape', lit: boolean }

      // Canvases for rendering pipeline
      this.overlayCanvas = null;
      this.overlayCtx = null;

      // Offscreen buffers for compositing shadows per-light cleanly
      this.ambientBuffer = document.createElement("canvas");
      this.ambientCtx = this.ambientBuffer.getContext("2d", { alpha: true });

      this.lightBuffer = document.createElement("canvas");
      this.lightCtx = this.lightBuffer.getContext("2d", { alpha: true });

      // Offscreen buffers for per-pixel sprite silhouette extraction & solid shadow wedges
      this.spriteMaskBuffer = document.createElement("canvas");
      this.spriteMaskCtx = this.spriteMaskBuffer.getContext("2d", {
        willReadFrequently: true,
      });

      this.spriteShadowBuffer = document.createElement("canvas");
      this.spriteShadowCtx = this.spriteShadowBuffer.getContext("2d", {
        alpha: true,
      });

      this.spriteRimBuffer = document.createElement("canvas");
      this.spriteRimCtx = this.spriteRimBuffer.getContext("2d", {
        alpha: true,
      });

      // Cache for skin image extraction so SVG/Bitmap costumes don't re-parse every frame
      this.skinCache = new Map();

      // Per-frame cache for extracted sprite silhouette masks so multiple lights
      // don't re-draw and re-read getImageData for the same sprite in the same frame!
      this.frameMaskCache = new Map();

      // Cached stage layout metrics to avoid synchronous getBoundingClientRect() reflows every frame
      this._lastSyncTime = 0;
      this._cachedRect = null;

      this._initOverlay();
      this._bindEvents();
    }

    /**
     * Initializes the HTML5 Canvas overlay directly on top of the TurboWarp stage.
     */
    _initOverlay() {
      if (!renderer || !renderer.canvas) return;

      const stageCanvas = renderer.canvas;
      const parent = stageCanvas.parentElement;

      // Remove existing overlay if reloading extension
      const existing = document.getElementById("tw-dynamic-lighting-overlay");
      if (existing) existing.remove();

      this.overlayCanvas = document.createElement("canvas");
      this.overlayCanvas.id = "tw-dynamic-lighting-overlay";
      this.overlayCanvas.style.position = "absolute";
      this.overlayCanvas.style.pointerEvents = "none";
      this.overlayCanvas.style.zIndex = "10";
      this.overlayCanvas.style.top = "0px";
      this.overlayCanvas.style.left = "0px";

      if (parent) {
        // Ensure parent can anchor absolute positioning if static
        const computedPos = window.getComputedStyle(parent).position;
        if (computedPos === "static") {
          parent.style.position = "relative";
        }
        parent.appendChild(this.overlayCanvas);
      }

      this.overlayCtx = this.overlayCanvas.getContext("2d", { alpha: true });
      this._syncCanvasSize(true);
    }

    /**
     * Keeps the overlay and offscreen buffers synchronized with the TurboWarp stage size.
     * Throttles DOM layout queries to prevent forced reflow bottlenecks at 60 FPS.
     */
    _syncCanvasSize(force = false) {
      if (!renderer || !renderer.canvas || !this.overlayCanvas) return;

      const stageCanvas = renderer.canvas;
      const now = performance.now();

      // Only query DOM bounding rects every ~250ms unless forced (resize/mode change)
      if (force || !this._cachedRect || now - this._lastSyncTime > 250) {
        const rect = stageCanvas.getBoundingClientRect();
        const parentRect = stageCanvas.parentElement
          ? stageCanvas.parentElement.getBoundingClientRect()
          : rect;

        this._cachedRect = {
          left: rect.left - parentRect.left,
          top: rect.top - parentRect.top,
          width: rect.width,
          height: rect.height,
        };
        this._lastSyncTime = now;

        this.overlayCanvas.style.left = `${this._cachedRect.left}px`;
        this.overlayCanvas.style.top = `${this._cachedRect.top}px`;
        this.overlayCanvas.style.width = `${this._cachedRect.width}px`;
        this.overlayCanvas.style.height = `${this._cachedRect.height}px`;
      }

      const stageWidth = runtime.stageWidth || 480;
      const stageHeight = runtime.stageHeight || 360;

      let targetW, targetH;
      if (this.renderMode === "pixelated") {
        const scale = Math.max(0.05, Math.min(1, this.pixelScale));
        targetW = Math.max(1, Math.floor(stageWidth * scale));
        targetH = Math.max(1, Math.floor(stageHeight * scale));
        if (this.overlayCanvas.style.imageRendering !== "pixelated") {
          this.overlayCanvas.style.imageRendering = "pixelated";
        }
      } else {
        // Cap smooth internal buffer at native stage resolution (or 1x CSS size up to 960x720)
        // for crisp visuals without wasting fill-rate on 4K Retina canvases
        const rawW = this._cachedRect.width || stageWidth;
        const rawH = this._cachedRect.height || stageHeight;
        const maxW = Math.max(stageWidth * 1.5, 720);
        const downscale = rawW > maxW ? maxW / rawW : 1;
        targetW = Math.max(1, Math.floor(rawW * downscale));
        targetH = Math.max(1, Math.floor(rawH * downscale));
        if (this.overlayCanvas.style.imageRendering !== "auto") {
          this.overlayCanvas.style.imageRendering = "auto";
        }
      }

      if (
        this.overlayCanvas.width !== targetW ||
        this.overlayCanvas.height !== targetH
      ) {
        this.overlayCanvas.width = targetW;
        this.overlayCanvas.height = targetH;
        this.ambientBuffer.width = targetW;
        this.ambientBuffer.height = targetH;
        this.lightBuffer.width = targetW;
        this.lightBuffer.height = targetH;
        this.spriteMaskBuffer.width = targetW;
        this.spriteMaskBuffer.height = targetH;
        this.spriteShadowBuffer.width = targetW;
        this.spriteShadowBuffer.height = targetH;

        const smooth = this.renderMode === "smooth";
        this.overlayCtx.imageSmoothingEnabled = smooth;
        this.ambientCtx.imageSmoothingEnabled = smooth;
        this.lightCtx.imageSmoothingEnabled = smooth;
        this.spriteMaskCtx.imageSmoothingEnabled = smooth;
        this.spriteShadowCtx.imageSmoothingEnabled = smooth;
      }
    }

    /**
     * Hooks into VM lifecycle events for 60 FPS rendering and clean resets.
     */
    _bindEvents() {
      // Render lighting after frame scripts run
      runtime.on("AFTER_EXECUTE", () => {
        this.render();
      });

      // Reset lights and obstacles when green flag clicked or project stops
      const resetAll = () => {
        this.lights.clear();
        this.obstacles.clear();
        this.frameMaskCache.clear();
        this.enabled = false;
        if (this.overlayCtx && this.overlayCanvas) {
          this.overlayCtx.clearRect(
            0,
            0,
            this.overlayCanvas.width,
            this.overlayCanvas.height
          );
        }
      };

      runtime.on("PROJECT_START", resetAll);
      runtime.on("PROJECT_STOP_ALL", resetAll);

      // Clean up deleted clones from obstacles map automatically
      runtime.on("targetWasRemoved", (target) => {
        if (target && target.id) {
          this.obstacles.delete(target.id);
        }
      });

      window.addEventListener("resize", () => this._syncCanvasSize(true));
    }

    /**
     * Dynamically locates the active Camera V2 extension instance on runtime or VM.
     */
    _findCameraV2Extension() {
      const directKeys = [
        "ext_SPcameraV2",
        "ext_spcamerav2",
        "ext_cameraV2",
        "ext_camerav2",
        "ext_DTcamera2",
        "ext_cameraControls",
        "camera2",
        "camera",
      ];
      for (const key of directKeys) {
        if (runtime[key]) return runtime[key];
      }

      for (const key of Object.keys(runtime)) {
        if (key.startsWith("ext_") && key.toLowerCase().includes("camera")) {
          return runtime[key];
        }
      }

      if (
        vm.extensionManager &&
        vm.extensionManager._loadedExtensions instanceof Map
      ) {
        for (const [extId, extInstance] of vm.extensionManager._loadedExtensions.entries()) {
          if (extId.toLowerCase().includes("camera") && extInstance) {
            return extInstance;
          }
        }
      }

      return null;
    }

    /**
     * Reads Camera v2 state if the extension is loaded and active in the VM.
     * Supports SharkPool's Camera V2 (multi-camera & default camera), Camera V1, and renderer camera state.
     */
    _getCameraTransform() {
      let camX = 0;
      let camY = 0;
      let zoom = 1;
      let rotation = 0; // degrees

      const camExt = this._findCameraV2Extension();

      if (camExt) {
        let camObj = null;

        const camCollection = camExt.cameras || camExt._cameras;
        if (camCollection instanceof Map) {
          const activeName =
            camExt.currentCamera || camExt.activeCamera || "default";
          camObj =
            camCollection.get(activeName) ||
            camCollection.get("default") ||
            camCollection.values().next().value;
        } else if (camCollection && typeof camCollection === "object") {
          const activeName =
            camExt.currentCamera || camExt.activeCamera || "default";
          camObj =
            camCollection[activeName] ||
            camCollection.default ||
            Object.values(camCollection)[0];
        } else {
          camObj = camExt;
        }

        if (camObj) {
          if (Array.isArray(camObj.pos)) {
            camX = Number(camObj.pos[0]) || 0;
            camY = Number(camObj.pos[1]) || 0;
          } else {
            camX = Number(camObj.x ?? camObj.camX ?? 0) || 0;
            camY = Number(camObj.y ?? camObj.camY ?? 0) || 0;
          }

          const rawZoom = Number(
            camObj.zoom ?? camObj.scale ?? camObj.size ?? 100
          );
          if (!isNaN(rawZoom) && rawZoom > 0) {
            if (
              camObj.zoom !== undefined &&
              camObj.scale === undefined &&
              !camObj._isNormalized
            ) {
              zoom = rawZoom / 100;
            } else {
              zoom = rawZoom > 10 ? rawZoom / 100 : rawZoom;
            }
          }

          if (camObj.rotation !== undefined) {
            rotation = Number(camObj.rotation) || 0;
          } else if (camObj.angle !== undefined) {
            rotation = Number(camObj.angle) || 0;
          } else if (camObj.dir !== undefined) {
            rotation = Number(camObj.dir) || 0;
          } else if (camObj.direction !== undefined) {
            const d = Number(camObj.direction);
            rotation = isNaN(d) ? 0 : d - 90;
          }
        }
      } else if (renderer && renderer._camera) {
        camX = Number(renderer._camera.x) || 0;
        camY = Number(renderer._camera.y) || 0;
        zoom = Number(renderer._camera.zoom) || 1;
        rotation = Number(renderer._camera.rotation) || 0;
      }

      if (
        camX === 0 &&
        camY === 0 &&
        zoom === 1 &&
        rotation === 0 &&
        renderer &&
        typeof renderer.getCameraPosition === "function"
      ) {
        const pos = renderer.getCameraPosition();
        if (Array.isArray(pos)) {
          camX = Number(pos[0]) || 0;
          camY = Number(pos[1]) || 0;
        }
      }

      return { camX, camY, zoom, rotation };
    }

    /**
     * Converts Scratch World Coordinates (0,0 center, +Y up) into
     * Buffer Canvas Coordinates (0,0 top-left, +Y down) taking Camera v2 into account.
     */
    _worldToCanvas(wx, wy, canvasW, canvasH, cam) {
      const stageW = runtime.stageWidth || 480;
      const stageH = runtime.stageHeight || 360;

      let rx = wx - cam.camX;
      let ry = wy - cam.camY;

      if (cam.rotation !== 0) {
        const rad = (-cam.rotation * Math.PI) / 180;
        const cos = Math.cos(rad);
        const sin = Math.sin(rad);
        const nx = rx * cos - ry * sin;
        const ny = rx * sin + ry * cos;
        rx = nx;
        ry = ny;
      }

      rx *= cam.zoom;
      ry *= cam.zoom;

      const scaleX = canvasW / stageW;
      const scaleY = canvasH / stageH;

      return {
        x: (rx + stageW / 2) * scaleX,
        y: (stageH / 2 - ry) * scaleY,
      };
    }

    /**
     * Retrieves or caches a drawable HTMLImageElement / HTMLCanvasElement for a Scratch costume.
     */
    _getCostumeDrawable(target) {
      if (!target || typeof target.getCurrentCostume !== "function") return null;
      const costume = target.getCurrentCostume();
      if (!costume) return null;

      const cacheKey = `${costume.assetId}_${costume.skinId}`;
      if (this.skinCache.has(cacheKey)) {
        const entry = this.skinCache.get(cacheKey);
        return entry.ready ? entry.source : null;
      }

      if (renderer && renderer._allSkins && costume.skinId !== undefined) {
        const skin = renderer._allSkins[costume.skinId];
        if (skin) {
          if (skin._svgImage && skin._svgImage.complete) {
            this.skinCache.set(cacheKey, {
              ready: true,
              source: skin._svgImage,
            });
            return skin._svgImage;
          }
          if (skin._canvas) {
            this.skinCache.set(cacheKey, {
              ready: true,
              source: skin._canvas,
            });
            return skin._canvas;
          }
        }
      }

      if (costume.asset && typeof costume.asset.encodeDataURI === "function") {
        const entry = { ready: false, source: null };
        this.skinCache.set(cacheKey, entry);

        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => {
          entry.ready = true;
          entry.source = img;
        };
        img.src = costume.asset.encodeDataURI();
      }

      return null;
    }

    /**
     * Extracts world-space geometry (polygon, circle, or sprite shape) for all active obstacles.
     */
    _getObstacleGeometries(canvasW, canvasH, cam) {
      const geometries = [];
      const stageW = runtime.stageWidth || 480;
      const scaleFactor = (canvasW / stageW) * cam.zoom;

      for (const [targetId, obs] of this.obstacles.entries()) {
        const target = obs.target;
        if (!target || !runtime.targets.includes(target) || !target.visible) {
          continue;
        }

        const bounds = target.getBounds();
        if (!bounds) continue;

        const centerWorldX = (bounds.left + bounds.right) / 2;
        const centerWorldY = (bounds.top + bounds.bottom) / 2;
        const worldW = Math.abs(bounds.right - bounds.left);
        const worldH = Math.abs(bounds.top - bounds.bottom);

        if (worldW <= 0 || worldH <= 0) continue;

        const p1 = this._worldToCanvas(bounds.left, bounds.top, canvasW, canvasH, cam);
        const p2 = this._worldToCanvas(bounds.right, bounds.top, canvasW, canvasH, cam);
        const p3 = this._worldToCanvas(bounds.right, bounds.bottom, canvasW, canvasH, cam);
        const p4 = this._worldToCanvas(bounds.left, bounds.bottom, canvasW, canvasH, cam);

        const minX = Math.min(p1.x, p2.x, p3.x, p4.x);
        const maxX = Math.max(p1.x, p2.x, p3.x, p4.x);
        const minY = Math.min(p1.y, p2.y, p3.y, p4.y);
        const maxY = Math.max(p1.y, p2.y, p3.y, p4.y);

        // Skip obstacles completely off-screen
        if (maxX < 0 || minX > canvasW || maxY < 0 || minY > canvasH) {
          continue;
        }

        const isLit = obs.lit === true;

        if (obs.shape === "sprite shape") {
          const centerCanvas = this._worldToCanvas(
            target.x,
            target.y,
            canvasW,
            canvasH,
            cam
          );
          geometries.push({
            type: "sprite",
            targetId,
            target,
            centerCanvas,
            boundsPoints: [p1, p2, p3, p4],
            aabb: { minX, minY, maxX, maxY },
            scaleFactor,
            cam,
            lit: isLit,
          });
        } else if (obs.shape === "circle") {
          const centerCanvas = this._worldToCanvas(
            centerWorldX,
            centerWorldY,
            canvasW,
            canvasH,
            cam
          );
          const radiusCanvas = (Math.max(worldW, worldH) / 2) * scaleFactor;
          geometries.push({
            type: "circle",
            x: centerCanvas.x,
            y: centerCanvas.y,
            r: radiusCanvas,
            aabb: { minX, minY, maxX, maxY },
            lit: isLit,
          });
        } else {
          geometries.push({
            type: "polygon",
            points: [p1, p2, p3, p4],
            aabb: { minX, minY, maxX, maxY },
            lit: isLit,
          });
        }
      }

      return geometries;
    }

    /**
     * Checks whether a 2D point lies strictly inside a convex polygon (works at any rotation).
     */
    _isPointInPolygon(px, py, pts) {
      let hasPos = false;
      let hasNeg = false;
      const n = pts.length;
      for (let i = 0; i < n; i++) {
        const a = pts[i];
        const b = pts[(i + 1) % n];
        const cross = (b.x - a.x) * (py - a.y) - (b.y - a.y) * (px - a.x);
        if (cross > 0) hasPos = true;
        if (cross < 0) hasNeg = true;
        if (hasPos && hasNeg) return false;
      }
      return true;
    }

    /**
     * Extracts and caches a compact local bounding-box alpha mask for a sprite once per frame.
     * In pixelated mode, supersamples the mask at 4x resolution so low-res pixel blocks don't
     * inflate the obstacle silhouette or shift its shadow boundary.
     */
    _getSpriteFrameMask(obs, canvasW, canvasH) {
      if (this.frameMaskCache.has(obs.targetId)) {
        return this.frameMaskCache.get(obs.targetId);
      }

      const target = obs.target;
      const costume = target.getCurrentCostume ? target.getCurrentCostume() : null;
      const drawable = this._getCostumeDrawable(target);

      if (!costume || !drawable) {
        this.frameMaskCache.set(obs.targetId, null);
        return null;
      }

      // Supersample mask in pixelated mode (e.g. 4x when pixelScale=0.25) so silhouette edges
      // match exact world coordinates without coarse pixel quantization jitter
      const maskScale =
        this.renderMode === "pixelated"
          ? Math.max(1, Math.min(4, Math.round(1 / Math.max(0.1, this.pixelScale))))
          : 1;

      const minX = Math.max(0, Math.floor(obs.aabb.minX) - 1);
      const minY = Math.max(0, Math.floor(obs.aabb.minY) - 1);
      const maxX = Math.min(canvasW, Math.ceil(obs.aabb.maxX) + 1);
      const maxY = Math.min(canvasH, Math.ceil(obs.aabb.maxY) + 1);

      const boxW = maxX - minX;
      const boxH = maxY - minY;
      if (boxW <= 0 || boxH <= 0) {
        this.frameMaskCache.set(obs.targetId, null);
        return null;
      }

      const hiW = boxW * maskScale;
      const hiH = boxH * maskScale;

      if (this.spriteMaskBuffer.width < hiW) this.spriteMaskBuffer.width = hiW;
      if (this.spriteMaskBuffer.height < hiH) this.spriteMaskBuffer.height = hiH;

      const maskCtx = this.spriteMaskCtx;
      maskCtx.save();
      maskCtx.imageSmoothingEnabled = false;
      maskCtx.clearRect(0, 0, hiW, hiH);
      maskCtx.translate(
        (obs.centerCanvas.x - minX) * maskScale,
        (obs.centerCanvas.y - minY) * maskScale
      );

      const spriteRotDeg = (target.direction - 90) - obs.cam.rotation;
      maskCtx.rotate((spriteRotDeg * Math.PI) / 180);

      let scaleX = (target.size / 100) * obs.scaleFactor * maskScale;
      let scaleY = (target.size / 100) * obs.scaleFactor * maskScale;

      if (target.rotationStyle === "left-right") {
        maskCtx.rotate((-spriteRotDeg * Math.PI) / 180);
        maskCtx.rotate((-obs.cam.rotation * Math.PI) / 180);
        const normDir = ((target.direction % 360) + 360) % 360;
        if (normDir > 180) scaleX = -scaleX;
      } else if (target.rotationStyle === "don't rotate") {
        maskCtx.rotate((-spriteRotDeg * Math.PI) / 180);
        maskCtx.rotate((-obs.cam.rotation * Math.PI) / 180);
      }

      const resDiv = costume.bitmapResolution || 1;
      scaleX /= resDiv;
      scaleY /= resDiv;

      maskCtx.scale(scaleX, scaleY);

      const rotCenterX = costume.rotationCenterX || 0;
      const rotCenterY = costume.rotationCenterY || 0;
      maskCtx.drawImage(drawable, -rotCenterX, -rotCenterY);
      maskCtx.restore();

      const imgData = maskCtx.getImageData(0, 0, hiW, hiH);

      const localCanvas = document.createElement("canvas");
      localCanvas.width = hiW;
      localCanvas.height = hiH;
      localCanvas.getContext("2d").putImageData(imgData, 0, 0);

      const step = Math.max(hiW, hiH) > 140 ? 2 : 1;

      const entry = {
        minX,
        minY,
        boxW,
        boxH,
        hiW,
        hiH,
        maskScale,
        data: imgData.data,
        canvas: localCanvas,
        step,
      };
      this.frameMaskCache.set(obs.targetId, entry);
      return entry;
    }

    /**
     * Extrudes solid, gap-free shadow wedges outward from (lightX, lightY) using cached sprite masks.
     */
    _castSpriteSilhouetteShadow(targetCtx, lightX, lightY, lightRadius, obs, canvasW, canvasH) {
      const mask = this._getSpriteFrameMask(obs, canvasW, canvasH);

      if (!mask) {
        this._castPolygonShadow(
          targetCtx,
          lightX,
          lightY,
          lightRadius,
          obs.boundsPoints,
          obs.lit
        );
        return;
      }

      const {
        minX,
        minY,
        boxW,
        boxH,
        hiW,
        hiH,
        maskScale,
        data,
        canvas: localCanvas,
        step,
      } = mask;
      const invMaskScale = 1 / maskScale;

      // 1. Check if light center sits directly inside an opaque pixel of this sprite
      const localLx = Math.round((lightX - minX) * maskScale);
      const localLy = Math.round((lightY - minY) * maskScale);
      if (localLx >= 0 && localLx < hiW && localLy >= 0 && localLy < hiH) {
        const centerAlpha = data[(localLy * hiW + localLx) * 4 + 3];
        if (centerAlpha >= 100) {
          if (!obs.lit) {
            targetCtx.save();
            targetCtx.globalCompositeOperation = "destination-out";
            targetCtx.globalAlpha = 0.85;
            targetCtx.drawImage(localCanvas, minX, minY, boxW, boxH);
            targetCtx.restore();
          }
          return;
        }
      }

      // 2. Reset pre-allocated Polar Angular Occlusion Bins
      BIN_OCCUPIED.fill(0);
      BIN_MIN_DIST.fill(Infinity);

      // Use exact sub-pixel sample radius so low-res pixelated mode doesn't inflate shadow cones
      const discRadius = step * invMaskScale * 0.65;
      let anyOccluded = false;

      for (let py = 0; py < hiH; py += step) {
        const rowOffset = py * hiW * 4;
        const worldPy = minY + (py + 0.5) * invMaskScale;
        const dy = worldPy - lightY;
        const dySq = dy * dy;

        for (let px = 0; px < hiW; px += step) {
          // Higher alpha threshold (100) avoids anti-aliased fringe pixels casting phantom shadows
          if (data[rowOffset + px * 4 + 3] < 100) continue;

          const worldPx = minX + (px + 0.5) * invMaskScale;
          const dx = worldPx - lightX;
          const distSq = dx * dx + dySq;
          if (distSq < 1.0) continue;

          const dist = Math.sqrt(distSq);
          const ratio = discRadius / dist;
          const halfSpan = ratio >= 1 ? Math.PI * 0.45 : Math.asin(ratio);
          const centerAngle = Math.atan2(dy, dx);

          const startAng = centerAngle - halfSpan;
          const endAng = centerAngle + halfSpan;

          const startBinFloat =
            (((startAng % TWO_PI) + TWO_PI) % TWO_PI) * INV_TWO_PI * NUM_BINS;
          const spanBins = Math.max(
            1,
            Math.ceil((endAng - startAng) * INV_TWO_PI * NUM_BINS)
          );
          const startBin = Math.floor(startBinFloat);

          for (let b = 0; b <= spanBins; b++) {
            const idx = (startBin + b) % NUM_BINS;
            BIN_OCCUPIED[idx] = 1;
            if (dist < BIN_MIN_DIST[idx]) {
              BIN_MIN_DIST[idx] = dist;
            }
          }
          anyOccluded = true;
        }
      }

      if (!anyOccluded) return;

      // 3. Render contiguous occluded angular runs as solid filled polygons using precomputed trig tables
      const shadCtx = this.spriteShadowCtx;
      shadCtx.save();
      shadCtx.globalCompositeOperation = "source-over";
      shadCtx.clearRect(0, 0, canvasW, canvasH);
      shadCtx.fillStyle = "rgba(0, 0, 0, 1)";

      const shadowRadius = lightRadius * 2.5 + 1000;

      let startOffset = 0;
      for (let i = 0; i < NUM_BINS; i++) {
        if (!BIN_OCCUPIED[i]) {
          startOffset = i;
          break;
        }
      }

      let i = 0;
      while (i < NUM_BINS) {
        const binIdx = (startOffset + i) % NUM_BINS;
        if (!BIN_OCCUPIED[binIdx]) {
          i++;
          continue;
        }

        const runStart = i;
        while (i < NUM_BINS && BIN_OCCUPIED[(startOffset + i) % NUM_BINS]) {
          i++;
        }
        const runEnd = i;

        shadCtx.beginPath();

        const firstIdx = (startOffset + runStart) % NUM_BINS;
        shadCtx.moveTo(
          lightX + COS_TABLE[firstIdx] * shadowRadius,
          lightY + SIN_TABLE[firstIdx] * shadowRadius
        );

        for (let k = runStart + 1; k <= runEnd; k++) {
          const idx = (startOffset + k) % NUM_BINS;
          shadCtx.lineTo(
            lightX + COS_TABLE[idx] * shadowRadius,
            lightY + SIN_TABLE[idx] * shadowRadius
          );
        }

        for (let k = runEnd - 1; k >= runStart; k--) {
          const idx = (startOffset + k) % NUM_BINS;
          const innerR = Math.max(0, BIN_MIN_DIST[idx] - 0.5);
          shadCtx.lineTo(
            lightX + COS_MID_TABLE[idx] * innerR,
            lightY + SIN_MID_TABLE[idx] * innerR
          );
        }

        shadCtx.closePath();
        shadCtx.fill();
      }

      // 4. Illumination behavior on the obstacle itself:
      // - If `lit` is true: erase the entire sprite silhouette from the shadow so the character/prop
      //   is fully lit by the light while still casting a full shadow behind it!
      // - If `lit` is false (default): fill the sprite silhouette completely inside the shadow buffer
      //   so zero light outline leaks onto the unlit obstacle.
      if (obs.lit) {
        shadCtx.globalCompositeOperation = "destination-out";
        shadCtx.drawImage(localCanvas, minX, minY, boxW, boxH);
      } else {
        shadCtx.globalCompositeOperation = "source-over";
        shadCtx.drawImage(localCanvas, minX, minY, boxW, boxH);
      }
      shadCtx.restore();

      // Carve the solid silhouette shadow out of the current light buffer
      targetCtx.drawImage(this.spriteShadowBuffer, 0, 0);
    }

    /**
     * Projects shadow quads behind back-facing edges of a convex polygon.
     */
    _castPolygonShadow(ctx, lightX, lightY, lightRadius, pts, lit = false) {
      if (this._isPointInPolygon(lightX, lightY, pts)) {
        if (!lit) {
          ctx.save();
          ctx.globalAlpha = 0.85;
          ctx.beginPath();
          ctx.moveTo(pts[0].x, pts[0].y);
          for (let i = 1; i < pts.length; i++) {
            ctx.lineTo(pts[i].x, pts[i].y);
          }
          ctx.closePath();
          ctx.fill();
          ctx.restore();
        }
        return;
      }

      const shadowLength = lightRadius * 2.5 + 1000;
      const n = pts.length;

      for (let i = 0; i < n; i++) {
        const a = pts[i];
        const b = pts[(i + 1) % n];

        const edgeDx = b.x - a.x;
        const edgeDy = b.y - a.y;

        const midX = (a.x + b.x) * 0.5;
        const midY = (a.y + b.y) * 0.5;
        const toLightX = lightX - midX;
        const toLightY = lightY - midY;

        const facing = edgeDy * toLightX - edgeDx * toLightY;

        if (facing < 0) {
          const dirAx = a.x - lightX;
          const dirAy = a.y - lightY;
          const lenA = Math.hypot(dirAx, dirAy) || 1;

          const dirBx = b.x - lightX;
          const dirBy = b.y - lightY;
          const lenB = Math.hypot(dirBx, dirBy) || 1;

          const extAx = a.x + (dirAx / lenA) * shadowLength;
          const extAy = a.y + (dirAy / lenA) * shadowLength;
          const extBx = b.x + (dirBx / lenB) * shadowLength;
          const extBy = b.y + (dirBy / lenB) * shadowLength;

          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.lineTo(extBx, extBy);
          ctx.lineTo(extAx, extAy);
          ctx.closePath();
          ctx.fill();
          ctx.stroke();
        }
      }

      // If marked as lit, keep the polygon's own surface illuminated while casting its shadow behind it
      if (lit) {
        ctx.save();
        ctx.globalCompositeOperation = "source-over";
        ctx.beginPath();
        ctx.moveTo(pts[0].x, pts[0].y);
        for (let i = 1; i < pts.length; i++) {
          ctx.lineTo(pts[i].x, pts[i].y);
        }
        ctx.closePath();
        // Restore light inside the polygon by re-filling with the active light gradient
        if (ctx._activeLightFill) {
          ctx.fillStyle = ctx._activeLightFill;
          ctx.fill();
        }
        ctx.restore();
      }
    }

    /**
     * Casts 2D shadow volumes away from a light source for nearby obstacles.
     */
    _applyShadowsToLightBuffer(ctx, lightX, lightY, lightRadius, obstacles, canvasW, canvasH) {
      const shadowLength = lightRadius * 2.5 + 1000;

      ctx.save();
      ctx.globalCompositeOperation = "destination-out";
      ctx.fillStyle = "rgba(0, 0, 0, 1)";
      ctx.strokeStyle = "rgba(0, 0, 0, 1)";
      ctx.lineWidth = this.renderMode === "pixelated" ? 0.5 : 1.25;
      ctx.lineJoin = "round";

      for (const obs of obstacles) {
        // Broad-phase AABB distance cull: skip obstacles completely outside this light's radius
        if (obs.aabb) {
          const closestX = Math.max(obs.aabb.minX, Math.min(lightX, obs.aabb.maxX));
          const closestY = Math.max(obs.aabb.minY, Math.min(lightY, obs.aabb.maxY));
          const dx = lightX - closestX;
          const dy = lightY - closestY;
          if (dx * dx + dy * dy > lightRadius * lightRadius) {
            continue;
          }
        }

        if (obs.type === "sprite") {
          this._castSpriteSilhouetteShadow(
            ctx,
            lightX,
            lightY,
            lightRadius,
            obs,
            canvasW,
            canvasH
          );
        } else if (obs.type === "polygon") {
          this._castPolygonShadow(
            ctx,
            lightX,
            lightY,
            lightRadius,
            obs.points,
            obs.lit
          );
        } else if (obs.type === "circle") {
          const dx = obs.x - lightX;
          const dy = obs.y - lightY;
          const dist = Math.hypot(dx, dy);

          if (dist <= obs.r) {
            if (!obs.lit) {
              ctx.save();
              ctx.globalAlpha = 0.85;
              ctx.beginPath();
              ctx.arc(obs.x, obs.y, obs.r, 0, Math.PI * 2);
              ctx.fill();
              ctx.restore();
            }
            continue;
          }

          if (dist - obs.r > lightRadius) continue;

          const angleToCenter = Math.atan2(dy, dx);
          const halfAngle = Math.asin(obs.r / dist);

          const t1Angle = angleToCenter - halfAngle;
          const t2Angle = angleToCenter + halfAngle;

          const tangentDist = Math.sqrt(dist * dist - obs.r * obs.r);
          const t1x = lightX + Math.cos(t1Angle) * tangentDist;
          const t1y = lightY + Math.sin(t1Angle) * tangentDist;
          const t2x = lightX + Math.cos(t2Angle) * tangentDist;
          const t2y = lightY + Math.sin(t2Angle) * tangentDist;

          const ext1x = lightX + Math.cos(t1Angle) * shadowLength;
          const ext1y = lightY + Math.sin(t1Angle) * shadowLength;
          const ext2x = lightX + Math.cos(t2Angle) * shadowLength;
          const ext2y = lightY + Math.sin(t2Angle) * shadowLength;

          const backStartAngle = Math.atan2(t1y - obs.y, t1x - obs.x);
          const backEndAngle = Math.atan2(t2y - obs.y, t2x - obs.x);

          ctx.beginPath();
          ctx.moveTo(t1x, t1y);
          ctx.lineTo(ext1x, ext1y);
          ctx.lineTo(ext2x, ext2y);
          ctx.lineTo(t2x, t2y);
          ctx.arc(obs.x, obs.y, obs.r, backEndAngle, backStartAngle, false);
          ctx.closePath();
          ctx.fill();
          ctx.stroke();

          if (obs.lit && ctx._activeLightFill) {
            ctx.save();
            ctx.globalCompositeOperation = "source-over";
            ctx.fillStyle = ctx._activeLightFill;
            ctx.beginPath();
            ctx.arc(obs.x, obs.y, obs.r, 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();
          }
        }
      }

      ctx.restore();
    }

    /**
     * Parses any hex or CSS color into { r, g, b } components.
     */
    _parseColor(hex) {
      if (typeof hex !== "string") return { r: 255, g: 255, b: 255 };
      let clean = hex.trim().replace(/^#/, "");
      if (clean.length === 3) {
        clean = clean
          .split("")
          .map((c) => c + c)
          .join("");
      }
      const num = parseInt(clean, 16);
      if (isNaN(num)) return { r: 255, g: 255, b: 255 };
      return {
        r: (num >> 16) & 255,
        g: (num >> 8) & 255,
        b: num & 255,
      };
    }

    /**
     * Renders the complete 2D lighting and shadow frame at 60 FPS.
     */
    render() {
      if (!this.overlayCanvas || !this.overlayCtx) {
        this._initOverlay();
        if (!this.overlayCanvas) return;
      }

      this._syncCanvasSize(false);

      const w = this.overlayCanvas.width;
      const h = this.overlayCanvas.height;
      const outCtx = this.overlayCtx;

      outCtx.clearRect(0, 0, w, h);

      if (!this.enabled) return;

      // Clear per-frame sprite mask cache
      this.frameMaskCache.clear();

      const stageW = runtime.stageWidth || 480;
      const cam = this._getCameraTransform();
      const obstacles =
        this.obstacles.size > 0 ? this._getObstacleGeometries(w, h, cam) : [];

      // 1. Fill ambient darkness layer
      const ambCtx = this.ambientCtx;
      ambCtx.save();
      ambCtx.globalCompositeOperation = "source-over";
      ambCtx.clearRect(0, 0, w, h);

      const ambRGB = this._parseColor(this.ambientColor);
      const ambAlpha = Math.max(0, Math.min(1, this.ambientDarkness / 100));
      ambCtx.fillStyle = `rgba(${ambRGB.r}, ${ambRGB.g}, ${ambRGB.b}, ${ambAlpha})`;
      ambCtx.fillRect(0, 0, w, h);
      ambCtx.restore();

      // Prepare main overlay for additive color tinting
      outCtx.save();

      // 2. Process each light source
      for (const light of this.lights.values()) {
        if (light.radius <= 0 || light.intensity <= 0) continue;

        // When followCamera is true (default), light is anchored in world space and moves
        // relative to Camera V2. When false, light ignores Camera V2 and stays pinned to screen.
        const activeCam =
          light.followCamera !== false
            ? cam
            : { camX: 0, camY: 0, zoom: 1, rotation: 0 };
        const pixelRatio = (w / stageW) * activeCam.zoom;

        const pos = this._worldToCanvas(light.x, light.y, w, h, activeCam);
        const scaledRadius = light.radius * pixelRatio;

        // Frustum culling: skip lights completely outside the viewport
        if (
          pos.x + scaledRadius < 0 ||
          pos.x - scaledRadius > w ||
          pos.y + scaledRadius < 0 ||
          pos.y - scaledRadius > h
        ) {
          continue;
        }

        const lCtx = this.lightCtx;
        lCtx.save();
        lCtx.globalCompositeOperation = "source-over";

        // Only clear the dirty rectangle around this light instead of the full screen
        const dirtyX = Math.max(0, Math.floor(pos.x - scaledRadius - 2));
        const dirtyY = Math.max(0, Math.floor(pos.y - scaledRadius - 2));
        const dirtyW = Math.min(w - dirtyX, Math.ceil(scaledRadius * 2 + 4));
        const dirtyH = Math.min(h - dirtyY, Math.ceil(scaledRadius * 2 + 4));
        lCtx.clearRect(dirtyX, dirtyY, dirtyW, dirtyH);

        const rgb = this._parseColor(light.color);
        const alpha = Math.max(0, Math.min(1, light.intensity / 100));

        const grad = lCtx.createRadialGradient(
          pos.x,
          pos.y,
          0,
          pos.x,
          pos.y,
          scaledRadius
        );
        if (light.type === "spotlight") {
          // Spotlights maintain stronger beam intensity along their throw so cast shadows stand out clearly
          grad.addColorStop(0, `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${alpha})`);
          grad.addColorStop(
            0.72,
            `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${alpha * 0.65})`
          );
          grad.addColorStop(1, `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0)`);
        } else {
          grad.addColorStop(0, `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${alpha})`);
          grad.addColorStop(
            0.6,
            `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${alpha * 0.45})`
          );
          grad.addColorStop(1, `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0)`);
        }

        lCtx._activeLightFill = grad;
        lCtx.fillStyle = grad;
        lCtx.beginPath();

        if (light.type === "spotlight") {
          const effectiveDir = light.direction - activeCam.rotation;
          const centerRad = ((effectiveDir - 90) * Math.PI) / 180;
          const halfSpreadRad = ((light.spread / 2) * Math.PI) / 180;

          lCtx.moveTo(pos.x, pos.y);
          lCtx.arc(
            pos.x,
            pos.y,
            scaledRadius,
            centerRad - halfSpreadRad,
            centerRad + halfSpreadRad
          );
          lCtx.closePath();
        } else {
          lCtx.arc(pos.x, pos.y, scaledRadius, 0, Math.PI * 2);
        }

        lCtx.fill();
        lCtx.restore();

        // Carve shadows out of this light's buffer
        if (light.castShadows && obstacles.length > 0) {
          this._applyShadowsToLightBuffer(
            lCtx,
            pos.x,
            pos.y,
            scaledRadius,
            obstacles,
            w,
            h
          );
        }

        lCtx._activeLightFill = null;

        // Blit only the dirty bounding region of the light into ambientBuffer and outCtx
        if (dirtyW > 0 && dirtyH > 0) {
          ambCtx.save();
          ambCtx.globalCompositeOperation = "destination-out";
          ambCtx.drawImage(
            this.lightBuffer,
            dirtyX,
            dirtyY,
            dirtyW,
            dirtyH,
            dirtyX,
            dirtyY,
            dirtyW,
            dirtyH
          );
          ambCtx.restore();

          outCtx.save();
          outCtx.globalCompositeOperation = "lighter";
          outCtx.globalAlpha = 0.65;
          outCtx.drawImage(
            this.lightBuffer,
            dirtyX,
            dirtyY,
            dirtyW,
            dirtyH,
            dirtyX,
            dirtyY,
            dirtyW,
            dirtyH
          );
          outCtx.restore();
        }
      }

      // 3. Composite the carved ambient darkness onto the final canvas
      outCtx.globalCompositeOperation = "source-over";
      outCtx.drawImage(this.ambientBuffer, 0, 0);
      outCtx.restore();
    }

    // =========================================================================
    // SCRATCH EXTENSION METADATA & BLOCK DEFINITIONS
    // =========================================================================

    getInfo() {
      return {
        id: "dynamicLighting2D",
        name: "2D Lighting",
        color1: "#F59E0B",
        color2: "#D97706",
        color3: "#B45309",
        blocks: [
          // --- Environment & Rendering ---
          {
            opcode: "enableLighting",
            blockType: Scratch.BlockType.COMMAND,
            text: "enable lighting [ENABLED]",
            arguments: {
              ENABLED: {
                type: Scratch.ArgumentType.STRING,
                menu: "onOffMenu",
                defaultValue: "on",
              },
            },
          },
          {
            opcode: "setAmbientDarkness",
            blockType: Scratch.BlockType.COMMAND,
            text: "set ambient darkness to [PERCENT]%",
            arguments: {
              PERCENT: {
                type: Scratch.ArgumentType.NUMBER,
                defaultValue: 85,
              },
            },
          },
          {
            opcode: "setAmbientColor",
            blockType: Scratch.BlockType.COMMAND,
            text: "set ambient color to [COLOR]",
            arguments: {
              COLOR: {
                type: Scratch.ArgumentType.COLOR,
                defaultValue: "#000011",
              },
            },
          },
          {
            opcode: "setRenderMode",
            blockType: Scratch.BlockType.COMMAND,
            text: "set rendering mode to [MODE]",
            arguments: {
              MODE: {
                type: Scratch.ArgumentType.STRING,
                menu: "renderModeMenu",
                defaultValue: "smooth",
              },
            },
          },
          {
            opcode: "setPixelScale",
            blockType: Scratch.BlockType.COMMAND,
            text: "set pixelated resolution scale to [SCALE]",
            arguments: {
              SCALE: {
                type: Scratch.ArgumentType.NUMBER,
                defaultValue: 0.25,
              },
            },
          },

          "---",

          // --- Creating & Deleting Lights ---
          {
            opcode: "createPointLight",
            blockType: Scratch.BlockType.COMMAND,
            text: "create point light named [NAME] at x: [X] y: [Y] radius: [RADIUS] color: [COLOR]",
            arguments: {
              NAME: {
                type: Scratch.ArgumentType.STRING,
                defaultValue: "torch",
              },
              X: {
                type: Scratch.ArgumentType.NUMBER,
                defaultValue: 0,
              },
              Y: {
                type: Scratch.ArgumentType.NUMBER,
                defaultValue: 0,
              },
              RADIUS: {
                type: Scratch.ArgumentType.NUMBER,
                defaultValue: 150,
              },
              COLOR: {
                type: Scratch.ArgumentType.COLOR,
                defaultValue: "#FFAA33",
              },
            },
          },
          {
            opcode: "createSpotlight",
            blockType: Scratch.BlockType.COMMAND,
            text: "create spotlight named [NAME] at x: [X] y: [Y] direction: [DIR] spread: [ANGLE] radius: [RADIUS] color: [COLOR]",
            arguments: {
              NAME: {
                type: Scratch.ArgumentType.STRING,
                defaultValue: "flashlight",
              },
              X: {
                type: Scratch.ArgumentType.NUMBER,
                defaultValue: 0,
              },
              Y: {
                type: Scratch.ArgumentType.NUMBER,
                defaultValue: 0,
              },
              DIR: {
                type: Scratch.ArgumentType.ANGLE,
                defaultValue: 90,
              },
              ANGLE: {
                type: Scratch.ArgumentType.NUMBER,
                defaultValue: 45,
              },
              RADIUS: {
                type: Scratch.ArgumentType.NUMBER,
                defaultValue: 220,
              },
              COLOR: {
                type: Scratch.ArgumentType.COLOR,
                defaultValue: "#FFFFFF",
              },
            },
          },
          {
            opcode: "deleteLight",
            blockType: Scratch.BlockType.COMMAND,
            text: "delete light [NAME]",
            arguments: {
              NAME: {
                type: Scratch.ArgumentType.STRING,
                defaultValue: "torch",
              },
            },
          },
          {
            opcode: "deleteAllLights",
            blockType: Scratch.BlockType.COMMAND,
            text: "delete all lights",
          },

          "---",

          // --- Modifying Lights ---
          {
            opcode: "setLightProperty",
            blockType: Scratch.BlockType.COMMAND,
            text: "set [PROPERTY] of light [NAME] to [VALUE]",
            arguments: {
              PROPERTY: {
                type: Scratch.ArgumentType.STRING,
                menu: "modifyPropMenu",
                defaultValue: "x",
              },
              NAME: {
                type: Scratch.ArgumentType.STRING,
                defaultValue: "torch",
              },
              VALUE: {
                type: Scratch.ArgumentType.NUMBER,
                defaultValue: 0,
              },
            },
          },
          {
            opcode: "changeLightProperty",
            blockType: Scratch.BlockType.COMMAND,
            text: "change [PROPERTY] of light [NAME] by [VALUE]",
            arguments: {
              PROPERTY: {
                type: Scratch.ArgumentType.STRING,
                menu: "modifyPropMenu",
                defaultValue: "x",
              },
              NAME: {
                type: Scratch.ArgumentType.STRING,
                defaultValue: "torch",
              },
              VALUE: {
                type: Scratch.ArgumentType.NUMBER,
                defaultValue: 10,
              },
            },
          },
          {
            opcode: "setLightColor",
            blockType: Scratch.BlockType.COMMAND,
            text: "set color of light [NAME] to [COLOR]",
            arguments: {
              NAME: {
                type: Scratch.ArgumentType.STRING,
                defaultValue: "torch",
              },
              COLOR: {
                type: Scratch.ArgumentType.COLOR,
                defaultValue: "#FFAA33",
              },
            },
          },
          {
            opcode: "setLightCastShadows",
            blockType: Scratch.BlockType.COMMAND,
            text: "set cast shadows of light [NAME] to [BOOL]",
            arguments: {
              NAME: {
                type: Scratch.ArgumentType.STRING,
                defaultValue: "torch",
              },
              BOOL: {
                type: Scratch.ArgumentType.STRING,
                menu: "yesNoMenu",
                defaultValue: "yes",
              },
            },
          },
          {
            opcode: "setLightFollowCamera",
            blockType: Scratch.BlockType.COMMAND,
            text: "set follow camera of light [NAME] to [BOOL]",
            arguments: {
              NAME: {
                type: Scratch.ArgumentType.STRING,
                defaultValue: "torch",
              },
              BOOL: {
                type: Scratch.ArgumentType.STRING,
                menu: "yesNoMenu",
                defaultValue: "yes",
              },
            },
          },

          "---",

          // --- Light Obstacles (Sprites) ---
          {
            opcode: "setSpriteObstacle",
            blockType: Scratch.BlockType.COMMAND,
            text: "set this sprite as light obstacle [BOOL] ([LIT])",
            arguments: {
              BOOL: {
                type: Scratch.ArgumentType.STRING,
                menu: "yesNoMenu",
                defaultValue: "yes",
              },
              LIT: {
                type: Scratch.ArgumentType.STRING,
                menu: "litMenu",
                defaultValue: "unlit",
              },
            },
          },
          {
            opcode: "setSpriteShadowShape",
            blockType: Scratch.BlockType.COMMAND,
            text: "set shadow shape of this sprite to [SHAPE]",
            arguments: {
              SHAPE: {
                type: Scratch.ArgumentType.STRING,
                menu: "shadowShapeMenu",
                defaultValue: "sprite shape",
              },
            },
          },

          "---",

          // --- Reporters & Booleans ---
          {
            opcode: "lightExists",
            blockType: Scratch.BlockType.BOOLEAN,
            text: "light [NAME] exists?",
            arguments: {
              NAME: {
                type: Scratch.ArgumentType.STRING,
                defaultValue: "torch",
              },
            },
          },
          {
            opcode: "getLightProperty",
            blockType: Scratch.BlockType.REPORTER,
            text: "[PROPERTY] of light [NAME]",
            arguments: {
              PROPERTY: {
                type: Scratch.ArgumentType.STRING,
                menu: "reportPropMenu",
                defaultValue: "x",
              },
              NAME: {
                type: Scratch.ArgumentType.STRING,
                defaultValue: "torch",
              },
            },
          },
        ],
        menus: {
          onOffMenu: {
            acceptReporters: true,
            items: ["on", "off"],
          },
          renderModeMenu: {
            acceptReporters: true,
            items: ["smooth", "pixelated"],
          },
          yesNoMenu: {
            acceptReporters: true,
            items: ["yes", "no"],
          },
          litMenu: {
            acceptReporters: true,
            items: ["unlit", "lit"],
          },
          shadowShapeMenu: {
            acceptReporters: true,
            items: ["sprite shape", "bounding box", "circle"],
          },
          modifyPropMenu: {
            acceptReporters: true,
            items: [
              "x",
              "y",
              "radius",
              "direction",
              "spread angle",
              "intensity",
            ],
          },
          reportPropMenu: {
            acceptReporters: true,
            items: ["x", "y", "radius", "direction", "color"],
          },
        },
      };
    }

    // =========================================================================
    // BLOCK IMPLEMENTATIONS
    // =========================================================================

    enableLighting(args) {
      const val = Scratch.Cast.toString(args.ENABLED).toLowerCase();
      this.enabled = val === "on" || val === "true" || val === "1";
      if (!this.enabled && this.overlayCtx && this.overlayCanvas) {
        this.overlayCtx.clearRect(
          0,
          0,
          this.overlayCanvas.width,
          this.overlayCanvas.height
        );
      }
    }

    setAmbientDarkness(args) {
      const pct = Scratch.Cast.toNumber(args.PERCENT);
      this.ambientDarkness = Math.max(0, Math.min(100, pct));
    }

    setAmbientColor(args) {
      this.ambientColor = Scratch.Cast.toString(args.COLOR);
    }

    setRenderMode(args) {
      const mode = Scratch.Cast.toString(args.MODE).toLowerCase();
      this.renderMode = mode === "pixelated" ? "pixelated" : "smooth";
      this._syncCanvasSize(true);
    }

    setPixelScale(args) {
      const scale = Scratch.Cast.toNumber(args.SCALE);
      this.pixelScale = Math.max(0.05, Math.min(1, scale || 0.25));
      this._syncCanvasSize(true);
    }

    createPointLight(args) {
      const name = Scratch.Cast.toString(args.NAME);
      if (!name) return;

      this.lights.set(name, {
        name,
        type: "point",
        x: Scratch.Cast.toNumber(args.X),
        y: Scratch.Cast.toNumber(args.Y),
        radius: Math.max(0, Scratch.Cast.toNumber(args.RADIUS)),
        direction: 90,
        spread: 360,
        intensity: 100,
        color: Scratch.Cast.toString(args.COLOR),
        castShadows: true,
        followCamera: true,
      });
    }

    createSpotlight(args) {
      const name = Scratch.Cast.toString(args.NAME);
      if (!name) return;

      this.lights.set(name, {
        name,
        type: "spotlight",
        x: Scratch.Cast.toNumber(args.X),
        y: Scratch.Cast.toNumber(args.Y),
        direction: Scratch.Cast.toNumber(args.DIR),
        spread: Math.max(0, Math.min(360, Scratch.Cast.toNumber(args.ANGLE))),
        radius: Math.max(0, Scratch.Cast.toNumber(args.RADIUS)),
        intensity: 100,
        color: Scratch.Cast.toString(args.COLOR),
        castShadows: true,
        followCamera: true,
      });
    }

    deleteLight(args) {
      const name = Scratch.Cast.toString(args.NAME);
      this.lights.delete(name);
    }

    deleteAllLights() {
      this.lights.clear();
    }

    setLightProperty(args) {
      const name = Scratch.Cast.toString(args.NAME);
      const light = this.lights.get(name);
      if (!light) return;

      const prop = Scratch.Cast.toString(args.PROPERTY).toLowerCase();
      const val = Scratch.Cast.toNumber(args.VALUE);

      switch (prop) {
        case "x":
          light.x = val;
          break;
        case "y":
          light.y = val;
          break;
        case "radius":
          light.radius = Math.max(0, val);
          break;
        case "direction":
          light.direction = val;
          break;
        case "spread angle":
          light.spread = Math.max(0, Math.min(360, val));
          break;
        case "intensity":
          light.intensity = Math.max(0, Math.min(100, val));
          break;
      }
    }

    changeLightProperty(args) {
      const name = Scratch.Cast.toString(args.NAME);
      const light = this.lights.get(name);
      if (!light) return;

      const prop = Scratch.Cast.toString(args.PROPERTY).toLowerCase();
      const delta = Scratch.Cast.toNumber(args.VALUE);

      switch (prop) {
        case "x":
          light.x += delta;
          break;
        case "y":
          light.y += delta;
          break;
        case "radius":
          light.radius = Math.max(0, light.radius + delta);
          break;
        case "direction":
          light.direction += delta;
          break;
        case "spread angle":
          light.spread = Math.max(0, Math.min(360, light.spread + delta));
          break;
        case "intensity":
          light.intensity = Math.max(0, Math.min(100, light.intensity + delta));
          break;
      }
    }

    setLightColor(args) {
      const name = Scratch.Cast.toString(args.NAME);
      const light = this.lights.get(name);
      if (!light) return;

      light.color = Scratch.Cast.toString(args.COLOR);
    }

    setLightCastShadows(args) {
      const name = Scratch.Cast.toString(args.NAME);
      const light = this.lights.get(name);
      if (!light) return;

      const val = Scratch.Cast.toString(args.BOOL).toLowerCase();
      light.castShadows = val === "yes" || val === "true" || val === "1";
    }

    setLightFollowCamera(args) {
      const name = Scratch.Cast.toString(args.NAME);
      const light = this.lights.get(name);
      if (!light) return;

      const val = Scratch.Cast.toString(args.BOOL).toLowerCase();
      light.followCamera = val === "yes" || val === "true" || val === "1";
    }

    setSpriteObstacle(args, util) {
      const target = util.target;
      if (!target || target.isStage) return;

      const val = Scratch.Cast.toString(args.BOOL).toLowerCase();
      const isObs = val === "yes" || val === "true" || val === "1";

      const litVal = Scratch.Cast.toString(args.LIT).toLowerCase();
      const isLit = litVal === "lit" || litVal === "yes" || litVal === "true" || litVal === "1";

      if (isObs) {
        const existing = this.obstacles.get(target.id);
        this.obstacles.set(target.id, {
          target,
          shape: existing ? existing.shape : "sprite shape",
          lit: isLit,
        });
      } else {
        this.obstacles.delete(target.id);
      }
    }

    setSpriteShadowShape(args, util) {
      const target = util.target;
      if (!target || target.isStage) return;

      const rawShape = Scratch.Cast.toString(args.SHAPE).toLowerCase();
      let shape = "sprite shape";
      if (rawShape === "circle") shape = "circle";
      else if (rawShape === "bounding box") shape = "bounding box";

      const existing = this.obstacles.get(target.id);
      if (existing) {
        existing.shape = shape;
      } else {
        this.obstacles.set(target.id, {
          target,
          shape,
          lit: false,
        });
      }
    }

    lightExists(args) {
      const name = Scratch.Cast.toString(args.NAME);
      return this.lights.has(name);
    }

    getLightProperty(args) {
      const name = Scratch.Cast.toString(args.NAME);
      const light = this.lights.get(name);
      if (!light) return "";

      const prop = Scratch.Cast.toString(args.PROPERTY).toLowerCase();
      switch (prop) {
        case "x":
          return light.x;
        case "y":
          return light.y;
        case "radius":
          return light.radius;
        case "direction":
          return light.direction;
        case "color":
          return light.color;
        default:
          return "";
      }
    }
  }

  Scratch.extensions.register(new DynamicLightingExtension());
})(Scratch);