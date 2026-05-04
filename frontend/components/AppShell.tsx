"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import {
  Boxes,
  Camera,
  ChevronDown,
  Cpu,
  FolderKanban,
  Gauge,
  HardDrive,
  Home,
  LogIn,
  LogOut,
  MemoryStick,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { api, clearToken, getToken, isPublicPath } from "@/lib/api";
import { formatEta, isActiveTask, taskStatusLabel, taskTypeLabel } from "@/lib/labels";
import { readTrackedTaskIds, TRACKED_TASKS_EVENT, writeTrackedTaskIds } from "@/lib/taskTracking";
import type { Task, User } from "@/lib/types";

const nav = [
  { href: "/", label: "新建项目", icon: Home },
  { href: "/camera", label: "实时视频", icon: Camera },
  { href: "/projects", label: "项目控制台", icon: FolderKanban }
];

const RESOURCE_REFRESH_MS = 2000;

type ResourcePayload = {
  cpu?: Record<string, unknown>;
  memory?: Record<string, unknown>;
  gpu?: Record<string, unknown>;
  workers?: Record<string, unknown>;
};

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [resources, setResources] = useState<ResourcePayload>({});
  const [tasks, setTasks] = useState<Task[]>([]);
  const [taskOpen, setTaskOpen] = useState(false);

  useEffect(() => {
    if (!getToken()) {
      setUser(null);
      setTasks([]);
      if (!isPublicPath(pathname)) router.replace("/login");
      return;
    }
    void api.me()
      .then(setUser)
      .catch(() => {
        setUser(null);
        setTasks([]);
        if (!isPublicPath(pathname)) router.replace("/login");
      });
  }, [pathname, router]);

  useEffect(() => {
    if (!user) {
      setResources({});
      return;
    }
    let cancelled = false;
    const load = () => {
      void api.resources()
        .then((data) => {
          if (!cancelled) setResources(data);
        })
        .catch(() => {
          if (!cancelled) setResources({ gpu: { available: false, message: "管理员权限可查看完整资源状态" } });
        });
    };
    load();
    const timer = window.setInterval(load, RESOURCE_REFRESH_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [user]);

  useEffect(() => {
    function refreshTrackedTasks() {
      if (!user) {
        setTasks([]);
        return;
      }

      const trackedIds = readTrackedTaskIds();
      const trackedRequests = trackedIds.map((id) => api.task(id).catch(() => null));
      const adminRequest = user.role === "admin"
        ? api.adminTasks().then((data) => data.tasks).catch(() => [])
        : Promise.resolve([] as Task[]);

      void Promise.all([Promise.all(trackedRequests), adminRequest]).then(([tracked, adminTasks]) => {
        const merged = new Map<string, Task>();
        for (const task of adminTasks) {
          if (isActiveTask(task)) merged.set(task.id, task);
        }
        for (const task of tracked) {
          if (task && isActiveTask(task)) merged.set(task.id, task);
        }
        const active = Array.from(merged.values()).sort((a, b) => {
          const aTime = new Date(a.started_at ?? a.created_at).getTime();
          const bTime = new Date(b.started_at ?? b.created_at).getTime();
          return bTime - aTime;
        });
        setTasks(active);
        const activeTrackedIds = tracked.filter((task): task is Task => Boolean(task && isActiveTask(task))).map((task) => task.id);
        if (trackedIds.length !== activeTrackedIds.length || trackedIds.some((id, index) => id !== activeTrackedIds[index])) {
          writeTrackedTaskIds(activeTrackedIds);
        }
      });
    }

    refreshTrackedTasks();
    window.addEventListener(TRACKED_TASKS_EVENT, refreshTrackedTasks);
    const timer = window.setInterval(refreshTrackedTasks, 2500);
    return () => {
      window.removeEventListener(TRACKED_TASKS_EVENT, refreshTrackedTasks);
      window.clearInterval(timer);
    };
  }, [user]);

  const routeLabel = useMemo(() => getRouteLabel(pathname), [pathname]);
  const activeTask = tasks[0];

  function logout() {
    clearToken();
    setUser(null);
    router.replace("/login");
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <Link className="brand" href="/">
          <span className="brand-mark"><Boxes size={20} /></span>
          <span>3DGS</span>
        </Link>
        <nav className="nav-list" aria-label="主导航">
          {nav.map((item) => {
            const Icon = item.icon;
            const active = pathname === item.href || (item.href !== "/" && pathname.startsWith(item.href));
            return (
              <Link className={`nav-link ${active ? "active" : ""}`} href={item.href} key={item.href}>
                <Icon size={18} />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>
        <div className="sidebar-footer">
          {user ? (
            <>
              <div className="user-card">
                <strong title={user.username}>{user.username}</strong>
                <span>{user.role === "admin" ? "管理员" : "普通用户"}</span>
              </div>
              <button className="ghost-button full" type="button" onClick={logout}>
                <LogOut size={16} />退出
              </button>
            </>
          ) : (
            <Link className="ghost-button full" href="/login"><LogIn size={16} />登录</Link>
          )}
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div className="top-task">
            {activeTask ? (
              <>
                <button className="top-task-summary" type="button" onClick={() => setTaskOpen((value) => !value)}>
                  <span className="activity-dot" />
                  <span className="top-task-title" title={activeTask.current_stage || activeTask.type}>
                    {taskTypeLabel(activeTask.type)} / {activeTask.current_stage || taskStatusLabel(activeTask.status)}
                  </span>
                  <span className="top-task-meta">
                    {Math.round(activeTask.progress || 0)}% · {formatEta(activeTask.eta_seconds)}
                    <ChevronDown size={14} />
                  </span>
                </button>
                {taskOpen ? (
                  <div className="task-popover">
                    {tasks.map((task) => (
                      <Link className="task-popover-row" href={`/projects/${task.project_id}`} key={task.id} onClick={() => setTaskOpen(false)}>
                        <span className="truncate">
                          <strong>{taskTypeLabel(task.type)}</strong> · {task.current_stage || taskStatusLabel(task.status)}
                        </span>
                        <span className={`status-pill ${task.status}`}>{Math.round(task.progress || 0)}%</span>
                        <div className="progress-track" style={{ gridColumn: "1 / -1" }} aria-label="任务进度">
                          <span style={{ width: `${Math.max(0, Math.min(100, task.progress || 0))}%` }} />
                        </div>
                      </Link>
                    ))}
                  </div>
                ) : null}
              </>
            ) : (
              <div className="breadcrumb">
                <span>WORKSPACE</span>
                <span>/</span>
                <strong>{routeLabel}</strong>
              </div>
            )}
          </div>

          <div className="resource-strip" aria-label="资源状态">
            <span className="resource-chip" title={resourceTitle(resources.cpu)}>
              <Cpu size={14} /><span>CPU</span><strong>{resourcePercent(resources.cpu?.usage_percent)}</strong>
            </span>
            <span className={`resource-chip ${resources.memory?.available ? "" : "warn"}`} title={resourceTitle(resources.memory)}>
              <MemoryStick size={14} /><span>RAM</span><strong>{memoryValue(resources.memory)}</strong>
            </span>
            <span className={`resource-chip ${resources.gpu?.available ? "" : "warn"}`} title={resourceTitle(resources.gpu)}>
              <Gauge size={14} /><span>GPU</span><strong>{gpuUsageValue(resources.gpu)}</strong>
            </span>
            <span className="resource-chip warn" title={resourceTitle(resources.gpu)}>
              <HardDrive size={14} /><span>VRAM</span><strong>{vramValue(resources.gpu)}</strong>
            </span>
          </div>
        </header>
        <main className="main-panel">{children}</main>
      </section>
    </div>
  );
}

function getRouteLabel(pathname: string): string {
  if (pathname === "/") return "新建工作流";
  if (pathname.startsWith("/upload")) return "离线上传与预览";
  if (pathname.startsWith("/projects/")) return "项目详情";
  if (pathname.startsWith("/projects")) return "项目控制台";
  if (pathname.startsWith("/camera")) return "实时视频";
  if (pathname.startsWith("/admin")) return "管理面板";
  if (pathname.startsWith("/feedback")) return "问题反馈";
  if (pathname.startsWith("/about")) return "算法合规";
  if (pathname.startsWith("/login")) return "账户登录";
  return "工作台";
}

function resourcePercent(value: unknown): string {
  const numeric = numericValue(value);
  if (numeric !== null) return `${Math.round(Math.max(0, Math.min(100, numeric)))}%`;
  if (typeof value === "string" && value.trim()) return value;
  return "--";
}

function gpuUsageValue(gpu?: Record<string, unknown>): string {
  if (!gpu?.available) return "--";
  const direct = numericValue(gpu.usage_percent);
  if (direct !== null) return resourcePercent(direct);
  const gpus = Array.isArray(gpu.gpus) ? gpu.gpus : [];
  const values = gpus
    .map((item) => numericValue((item as Record<string, unknown>).usage_percent))
    .filter((value): value is number => value !== null);
  return values.length ? resourcePercent(Math.max(...values)) : "--";
}

function vramValue(gpu?: Record<string, unknown>): string {
  if (!gpu?.available) return "--";
  const used = numericValue(gpu.memory_used);
  const total = numericValue(gpu.memory_total);
  if (used !== null && total !== null && total > 0) {
    return `${(used / 1024).toFixed(1)}G/${(total / 1024).toFixed(0)}G`;
  }
  const gpus = Array.isArray(gpu.gpus) ? gpu.gpus : [];
  const totals = gpus.map((item) => numericValue((item as Record<string, unknown>).memory_total) ?? 0);
  const usedValues = gpus.map((item) => numericValue((item as Record<string, unknown>).memory_used) ?? 0);
  const totalFromGpus = totals.reduce((sum, value) => sum + value, 0);
  const usedFromGpus = usedValues.reduce((sum, value) => sum + value, 0);
  if (totalFromGpus > 0) return `${(usedFromGpus / 1024).toFixed(1)}G/${(totalFromGpus / 1024).toFixed(0)}G`;
  const percent = numericValue(gpu.memory_usage_percent);
  if (percent !== null) return `${Math.round(percent)}%`;
  return "--";
}

function memoryValue(memory?: Record<string, unknown>): string {
  if (!memory?.available) return "--";
  const used = numericValue(memory.used);
  const total = numericValue(memory.total);
  if (used !== null && total !== null && total > 0) {
    return `${formatBinaryBytes(used)}/${formatBinaryBytes(total)}`;
  }
  return resourcePercent(memory.usage_percent);
}

function resourceTitle(resource?: Record<string, unknown>): string | undefined {
  if (!resource) return undefined;
  const parts = [
    typeof resource.source === "string" ? `source: ${resource.source}` : null,
    typeof resource.message === "string" && resource.message.trim() ? resource.message : null
  ].filter(Boolean);
  return parts.length ? parts.join("\n") : undefined;
}

function formatBinaryBytes(value: number): string {
  if (value < 1024 * 1024 * 1024) return `${(value / 1024 / 1024).toFixed(0)}M`;
  return `${(value / 1024 / 1024 / 1024).toFixed(1)}G`;
}

function numericValue(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim()) {
    const parsed = Number.parseFloat(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}
