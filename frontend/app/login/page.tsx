"use client";

import { LogIn, UserPlus } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { api, setToken } from "@/lib/api";

export default function LoginPage() {
  const router = useRouter();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("admin123");
  const [email, setEmail] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    setBusy(true);
    setError(null);
    try {
      const result = mode === "login"
        ? await api.login({ username, password })
        : await api.register({ username, password, email: email || undefined });
      setToken(result.access_token);
      router.replace("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "认证失败");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page auth-page">
      <section className="panel padded auth-panel stack">
        <div>
          <p className="eyebrow">Account</p>
          <h1>{mode === "login" ? "登录工作台" : "注册账号"}</h1>
          <p className="muted">使用真实后端账号进入重建工作台。开发环境默认管理员通常是 admin / admin123，部署时请通过环境变量替换。</p>
        </div>
        <div className="segmented">
          <button className={mode === "login" ? "active" : ""} type="button" onClick={() => setMode("login")}>登录</button>
          <button className={mode === "register" ? "active" : ""} type="button" onClick={() => setMode("register")}>注册</button>
        </div>
        <div className="field">
          <label>用户名</label>
          <input className="input" value={username} onChange={(event) => setUsername(event.target.value)} autoComplete="username" />
        </div>
        {mode === "register" ? (
          <div className="field">
            <label>邮箱</label>
            <input className="input" value={email} onChange={(event) => setEmail(event.target.value)} autoComplete="email" />
          </div>
        ) : null}
        <div className="field">
          <label>密码</label>
          <input className="input" value={password} onChange={(event) => setPassword(event.target.value)} type="password" autoComplete={mode === "login" ? "current-password" : "new-password"} />
        </div>
        <button className="button" type="button" onClick={() => void submit()} disabled={busy || !username || !password}>
          {mode === "login" ? <LogIn size={17} /> : <UserPlus size={17} />}
          {mode === "login" ? "登录" : "创建账号"}
        </button>
        {error ? <div className="error-box">{error}</div> : null}
      </section>
    </div>
  );
}
