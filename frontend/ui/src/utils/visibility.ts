type VisibilityCallback = (visible: boolean) => void;
interface ObserverGroup {
  observer: IntersectionObserver;
  callbacks: Map<Element, Set<VisibilityCallback>>;
}
const groups = new WeakMap<Element | Document, Map<string, ObserverGroup>>();

export function observeVisibility(
  element: Element,
  callback: VisibilityCallback,
  root: Element | Document | null = null,
  rootMargin = '0px',
): () => void {
  const key = root ?? document;
  let pool = groups.get(key);
  if (!pool) { pool = new Map(); groups.set(key, pool); }
  let group = pool.get(rootMargin);
  if (!group) {
    const callbacks = new Map<Element, Set<VisibilityCallback>>();
    const observer = new IntersectionObserver(entries => {
      for (const entry of entries) {
        for (const listener of callbacks.get(entry.target) ?? []) listener(entry.isIntersecting);
      }
    }, { root, rootMargin });
    group = { observer, callbacks };
    pool.set(rootMargin, group);
  }
  let listeners = group.callbacks.get(element);
  if (!listeners) {
    listeners = new Set(); group.callbacks.set(element, listeners);
    group.observer.observe(element);
  }
  listeners.add(callback);
  return () => {
    listeners!.delete(callback);
    if (!listeners!.size) { group!.observer.unobserve(element); group!.callbacks.delete(element); }
    if (!group!.callbacks.size) {
      group!.observer.disconnect(); pool!.delete(rootMargin);
      if (!pool!.size) groups.delete(key);
    }
  };
}
