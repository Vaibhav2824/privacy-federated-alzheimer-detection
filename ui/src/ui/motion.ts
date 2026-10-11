/**
 * Page motion, driven by GSAP and opted into with data attributes so the
 * markup helpers stay plain strings:
 *
 *   data-reveal         fade and rise into view on scroll
 *   data-stagger        reveal direct children one after another
 *   data-count          count every number in the element's text up from zero
 *   data-bar            grow a bar from zero to its inline width
 *   data-ring="0.87"    draw an SVG circle's stroke to that fraction
 *   .spotlight          cursor-following glow (CSS reads --mx / --my)
 *
 * Final values are always in the markup, so with reduced motion, or if GSAP
 * fails, every number and bar is simply shown as rendered.
 */

import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

const NUMBER = /\d+(?:\.\d+)?/g;

const reducedMotion = (): boolean => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/** Tween each number inside `el` from zero, keeping its decimals and the text around it. */
function countUp(el: HTMLElement, trigger: gsap.DOMTarget | undefined, delay: number): void {
  const template = el.textContent ?? '';
  const targets = template.match(NUMBER);
  if (!targets || template.includes(',')) return;
  const decimals = targets.map((t) => t.split('.')[1]?.length ?? 0);
  const draw = (p: number): void => {
    let i = 0;
    el.textContent = template.replace(NUMBER, () => {
      const value = Number(targets[i]) * p;
      return value.toFixed(decimals[i++]);
    });
  };
  const state = { p: 0 };
  draw(0);
  gsap.to(state, {
    p: 1,
    duration: 1.8,
    delay,
    ease: 'expo.out',
    onUpdate: () => draw(state.p),
    onComplete: () => { el.textContent = template; },
    scrollTrigger: trigger ? { trigger, start: 'top 90%', once: true } : undefined,
  });
}

function spotlight(el: HTMLElement): void {
  el.addEventListener('pointermove', (event) => {
    const box = el.getBoundingClientRect();
    el.style.setProperty('--mx', `${event.clientX - box.left}px`);
    el.style.setProperty('--my', `${event.clientY - box.top}px`);
  });
}

/**
 * Wire motion for everything under `root`. With `immediate`, animations start
 * now rather than on scroll, for content swapped in after a user action.
 */
export function animate(root: ParentNode = document, immediate = false): void {
  root.querySelectorAll<HTMLElement>('.spotlight').forEach(spotlight);
  if (reducedMotion()) return;
  const on = (el: Element): gsap.DOMTarget | undefined => (immediate ? undefined : el);

  root.querySelectorAll<HTMLElement>('[data-reveal]').forEach((el) => {
    gsap.from(el, {
      y: 28, opacity: 0, duration: 0.9, ease: 'power3.out',
      scrollTrigger: immediate ? undefined : { trigger: el, start: 'top 90%', once: true },
    });
  });

  root.querySelectorAll<HTMLElement>('[data-stagger]').forEach((el) => {
    gsap.from(el.children, {
      y: 24, opacity: 0, scale: 0.98, duration: 0.8, ease: 'power3.out', stagger: 0.08,
      scrollTrigger: immediate ? undefined : { trigger: el, start: 'top 90%', once: true },
    });
  });

  root.querySelectorAll<HTMLElement>('[data-count]').forEach((el) => countUp(el, on(el), Number(el.dataset.count) || 0));

  root.querySelectorAll<HTMLElement>('[data-bar]').forEach((el) => {
    gsap.from(el, {
      width: 0, duration: 1.4, ease: 'expo.out', delay: Number(el.dataset.bar) || 0.1,
      scrollTrigger: immediate ? undefined : { trigger: el, start: 'top 95%', once: true },
    });
  });

  root.querySelectorAll<SVGCircleElement>('circle[data-ring]').forEach((el) => {
    const length = el.getTotalLength();
    gsap.fromTo(el, { strokeDashoffset: length }, {
      strokeDashoffset: length * (1 - Number(el.dataset.ring)),
      duration: 2, ease: 'expo.out', delay: 0.2,
      scrollTrigger: immediate ? undefined : { trigger: el, start: 'top 95%', once: true },
    });
  });
}

/** A circular gauge; the stroke is drawn to `value` (0..1) by `animate`. */
export function ring(value: number, size = 220, stroke = 14, color = 'url(#ring-grad)', marker?: number): string {
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const v = Math.max(0, Math.min(1, value));
  const tick = marker === undefined ? '' : (() => {
    const a = marker * 2 * Math.PI - Math.PI / 2;
    const [x1, y1] = [size / 2 + (r - stroke) * Math.cos(a), size / 2 + (r - stroke) * Math.sin(a)];
    const [x2, y2] = [size / 2 + (r + stroke) * Math.cos(a), size / 2 + (r + stroke) * Math.sin(a)];
    return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#0f172a" stroke-width="2" stroke-dasharray="3 2"/>`;
  })();
  return `<svg viewBox="0 0 ${size} ${size}" class="h-full w-full" aria-hidden="true">
    <defs><linearGradient id="ring-grad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#14b8a6"/><stop offset="1" stop-color="#0ea5e9"/></linearGradient></defs>
    <circle cx="${size / 2}" cy="${size / 2}" r="${r}" fill="none" stroke="#e2e8f0" stroke-width="${stroke}"/>
    <circle cx="${size / 2}" cy="${size / 2}" r="${r}" fill="none" stroke="${color}" stroke-width="${stroke}" stroke-linecap="round"
      transform="rotate(-90 ${size / 2} ${size / 2})" stroke-dasharray="${c}" stroke-dashoffset="${c * (1 - v)}" data-ring="${v}"/>
    ${tick}
  </svg>`;
}
