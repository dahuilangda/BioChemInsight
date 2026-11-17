import { useCallback, useLayoutEffect, useRef } from 'react';

export function useEventCallback<Args extends unknown[], Result>(callback: (...args: Args) => Result) {
  const current = useRef(callback);
  useLayoutEffect(() => { current.current = callback; });
  return useCallback((...args: Args): Result => current.current(...args), []);
}
