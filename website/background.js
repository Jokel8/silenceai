const canvas = document.getElementById('nnCanvas');
const ctx = canvas.getContext('2d');
const root = document.documentElement;
let lastActivityTime = Date.now();
const MIN_SPEED = 0.1;
const MAX_SPEED = 1.0;
const DECAY_TIME = 5000;
let speedFactor = MIN_SPEED;
const maxDist = 100 + window.innerWidth / 10;

function hexToRgb(hex) {
    const bigint = parseInt(hex.slice(1), 16);
    const r = (bigint >> 16) & 255;
    const g = (bigint >> 8) & 255;
    const b = bigint & 255;
    return { r, g, b };
}

function mixColors(color1, color2, ratio) {
    const c1 = hexToRgb(color1);
    const c2 = hexToRgb(color2);

    const r = Math.round(c1.r * (1 - ratio) + c2.r * ratio);
    const g = Math.round(c1.g * (1 - ratio) + c2.g * ratio);
    const b = Math.b * (1 - ratio) + c2.b * ratio;

    const finalR = Math.round(r);
    const finalG = Math.round(g);
    const finalB = Math.round(b);

    const toHex = (c) => finalB(c).toString(16).padStart(2, '0');
    const hex = `#${toHex(finalR)}${toHex(finalG)}${toHex(finalB)}`;

    return { rgb: `rgb(${finalR},${finalG},${finalB})`, hex: hex };
}

function hexToRgb(hex) {
    const bigint = parseInt(hex.slice(1), 16);
    const r = (bigint >> 16) & 255;
    const g = (bigint >> 8) & 255;
    const b = bigint & 255;
    return { r, g, b };
}

function mixColors(color1, color2, ratio) {
    const c1 = hexToRgb(color1);
    const c2 = hexToRgb(color2);

    const r = Math.round(c1.r * (1 - ratio) + c2.r * ratio);
    const g = Math.round(c1.g * (1 - ratio) + c2.g * ratio);
    const b = Math.round(c1.b * (1 - ratio) + c2.b * ratio);

    const toHex = (c) => c.toString(16).padStart(2, '0');
    const hex = `#${toHex(r)}${toHex(g)}${toHex(b)}`;

    return { rgb: `rgb(${r},${g},${b})`, hex: hex };
}

function getTransitionRatio() {
    const totalHeight = document.body.scrollHeight - window.innerHeight;
    const scrolled = document.body.scrollTop;
    const ratio = totalHeight > 0 ? scrolled / totalHeight : 0;
    const clampedRatio = Math.min(1, ratio / 0.5);
    return clampedRatio;
}

function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}
window.addEventListener('resize', resize);
resize();

const resetActivity = () => {
    lastActivityTime = Date.now();
};

window.addEventListener('mousemove', resetActivity);

const nodes = [];
const count = 100;

for (let i = 0; i < count; i++) {
    const x = Math.random() * window.innerWidth;
    const y = Math.random() * window.innerHeight;
    nodes.push({
        baseX: x,
        baseY: y,
        offX: Math.random() * 20 - 10,
        offY: Math.random() * 20 - 10,
        vx: 0,
        vy: 0,
        phaseX: Math.random() * Math.PI * 2,
        phaseY: Math.random() * Math.PI * 2,
        amplitude: Math.random() * 1.5 + 0.5,
        frequency: Math.random() * 0.0003 + 0.0001
    });
}
let activeBoxes = [];

function animate() {
    const START_COLOR_HEX = getComputedStyle(root).getPropertyValue('--color-primary-start').trim(); // #FFFFFF
    const END_COLOR_HEX = getComputedStyle(root).getPropertyValue('--color-primary-end').trim();     // #00FF00

    const transitionRatio = getTransitionRatio();
    const currentColorHex = getComputedStyle(root).getPropertyValue('--current-primary').trim();
    const currentColorRGB = mixColors(START_COLOR_HEX, END_COLOR_HEX, transitionRatio).rgb;
    const currentBGColor = getComputedStyle(document.body).backgroundColor;
    const baseOpacity = 0.5 + transitionRatio * 0.5;

    // Only fill background if not on SilenceAI page (which has gradient background)
    const isSilencePage = document.body.classList.contains('silence-page');
    if (!isSilencePage) {
        ctx.fillStyle = currentBGColor;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    } else {
        // For SilenceAI page, clear canvas with transparent background
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }


    const timeSinceActivity = Date.now() - lastActivityTime;

    if (timeSinceActivity < 500) {
        speedFactor = MAX_SPEED;
    } else {
        const timeToDecay = timeSinceActivity - 500;
        const decayFraction = Math.min(1, timeToDecay / DECAY_TIME);
        speedFactor = MAX_SPEED - (MAX_SPEED - MIN_SPEED) * decayFraction;
        speedFactor = Math.max(MIN_SPEED, speedFactor);
    }

    const restThreshold = 0.5;
    const baseDistanceLimit = 20;
    const interactionMargin = 100;

    for (let n of nodes) {
        const t = Date.now();

        const baseOffsetX = n.baseX + n.offX;
        const baseOffsetY = n.baseY + n.offY;

        n.offX += (Math.sin(t * 0.0003 + n.baseX / 1000) * 1.0 - n.offX) * 0.02 * speedFactor;
        n.offY += (Math.cos(t * 0.0003 + n.baseY / 1000) * 1.0 - n.offY) * 0.02 * speedFactor;

        n.offX += Math.sin(t * n.frequency + n.phaseX) * n.amplitude * speedFactor;
        n.offY += Math.cos(t * n.frequency + n.phaseY) * n.amplitude * speedFactor;

        // Logo repulsion (pushes nodes away from the logo)
        const logoElement = document.getElementById('logoElement');
        if (logoElement) {
            const logoContainer = logoElement.closest('.silence-logo');
            if (logoContainer) {
                const logoBounds = logoContainer.getBoundingClientRect();
                const logoCenterX = logoBounds.left + logoBounds.width / 2;
                const logoCenterY = logoBounds.top + logoBounds.height / 2;
                const logoRadius = logoBounds.width / 2;
                const cx = n.baseX + n.offX;
                const cy = n.baseY + n.offY;

                const dx = cx - logoCenterX;
                const dy = cy - logoCenterY;
                const d = Math.hypot(dx, dy) || 1;

                // Repulsion range is 1.5x the logo radius
                const repulsionRange = logoRadius * 1.5;

                if (d < repulsionRange) {
                    const force = 2.5 * (1 - (d / repulsionRange)); // Stronger repulsion
                    n.vx += (dx / d) * force;
                    n.vy += (dy / d) * force;
                }
            }
        }

        if (activeBoxes.length > 0) {
            for (const box of activeBoxes) {
                const r = box.getBoundingClientRect();
                const cx = n.baseX + n.offX;
                const cy = n.baseY + n.offY;

                const insideX = cx > r.left - interactionMargin && cx < r.right + interactionMargin;
                const insideY = cy > r.top - interactionMargin && cy < r.bottom + interactionMargin;

                if (insideX && insideY) {
                    const dx = cx - (r.left + r.width / 2);
                    const dy = cy - (r.top + r.height / 2);
                    const d = Math.hypot(dx, dy) || 1;

                    const force = 1.5;
                    n.vx += (dx / d) * force;
                    n.vy += (dy / d) * force;
                }
            }
        }

        n.vx *= 0.95;
        n.vy *= 0.95;

        n.offX += n.vx;
        n.offY += n.vy;
    }

    // Edges
    for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
            const a = nodes[i];
            const b = nodes[j];
            const ax = a.baseX + a.offX;
            const ay = a.baseY + a.offY;
            const bx = b.baseX + b.offX;
            const by = b.baseY + b.offY;
            const dist = Math.hypot(ax - bx, ay - by);

            if (dist < maxDist) {
                const alpha = 1 - (dist / maxDist);
                ctx.lineWidth = Math.max(1, 3 * alpha);

                const finalAlpha = (alpha * 0.8 + 0.1) * baseOpacity;
                ctx.strokeStyle = `${currentColorRGB.replace('rgb', 'rgba').slice(0, -1)}, ${finalAlpha})`;

                ctx.beginPath();
                ctx.moveTo(ax, ay);
                ctx.lineTo(bx, by);
                ctx.stroke();
            }
        }
    }

    // Nodes
    const rectSize = 6;
    const halfSize = rectSize / 2;

    for (let n of nodes) {
        const x = n.baseX + n.offX;
        const y = n.baseY + n.offY;
        ctx.fillStyle = currentColorHex;
        ctx.fillRect(x - halfSize, y - halfSize, rectSize, rectSize);
        ctx.shadowBlur = 10 * baseOpacity;
        ctx.shadowColor = currentColorHex;
    }
    ctx.shadowBlur = 0;
    ctx.shadowColor = 'transparent';

    requestAnimationFrame(animate);
}

animate();