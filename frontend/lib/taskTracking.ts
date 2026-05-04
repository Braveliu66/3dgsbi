const TRACKED_TASKS_KEY = "three_dgs_tracked_tasks";
export const TRACKED_TASKS_EVENT = "three-dgs-tracked-tasks";

export function readTrackedTaskIds(): string[] {
  if (typeof window === "undefined") return [];
  try {
    const parsed = JSON.parse(window.localStorage.getItem(TRACKED_TASKS_KEY) ?? "[]");
    return Array.isArray(parsed) ? parsed.filter((item): item is string => typeof item === "string") : [];
  } catch {
    return [];
  }
}

export function writeTrackedTaskIds(ids: string[]): void {
  if (typeof window === "undefined") return;
  const unique = Array.from(new Set(ids)).slice(0, 12);
  window.localStorage.setItem(TRACKED_TASKS_KEY, JSON.stringify(unique));
  window.dispatchEvent(new Event(TRACKED_TASKS_EVENT));
}

export function rememberTaskId(id: string): void {
  writeTrackedTaskIds([id, ...readTrackedTaskIds()]);
}
