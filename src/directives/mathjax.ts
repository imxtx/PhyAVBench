import type { Directive } from 'vue'

type MathJaxWindow = {
  startup?: {
    promise?: Promise<unknown>
  }
  typesetPromise?: (elements?: HTMLElement[]) => Promise<unknown>
}

async function typesetElement(el: HTMLElement): Promise<void> {
  const mathJax = (window as Window & { MathJax?: MathJaxWindow }).MathJax
  if (!mathJax?.typesetPromise) {
    return
  }

  if (mathJax.startup?.promise) {
    await mathJax.startup.promise
  }

  await mathJax.typesetPromise([el])
}

function scheduleTypeset(el: HTMLElement): void {
  requestAnimationFrame(() => {
    void typesetElement(el)
  })
}

export const mathjaxDirective: Directive<HTMLElement, void> = {
  mounted(el) {
    scheduleTypeset(el)
  },
  updated(el) {
    scheduleTypeset(el)
  },
}
