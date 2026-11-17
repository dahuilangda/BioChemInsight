export type Region = { x: number; y: number; w: number; h: number };
export type Point = { x: number; y: number };
export type RegionAction = 'new' | 'move' | 'nw' | 'ne' | 'sw' | 'se';
const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));
export function adjustRegion(
  action: RegionAction,
  start: Point,
  end: Point,
  initial?: Region,
): Region {
  if (action === 'new' || !initial)
    return {
      x: Math.min(start.x, end.x),
      y: Math.min(start.y, end.y),
      w: Math.abs(end.x - start.x),
      h: Math.abs(end.y - start.y),
    };
  const dx = end.x - start.x,
    dy = end.y - start.y;
  if (action === 'move')
    return {
      ...initial,
      x: clamp(initial.x + dx, 0, 1 - initial.w),
      y: clamp(initial.y + dy, 0, 1 - initial.h),
    };
  const fixed = {
    x: action.includes('w') ? initial.x + initial.w : initial.x,
    y: action.includes('n') ? initial.y + initial.h : initial.y,
  };
  const moving = {
    x: clamp((action.includes('w') ? initial.x : initial.x + initial.w) + dx, 0, 1),
    y: clamp((action.includes('n') ? initial.y : initial.y + initial.h) + dy, 0, 1),
  };
  return adjustRegion('new', fixed, moving);
}
