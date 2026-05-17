"use client";

import { RotateCcw, Save, SlidersHorizontal } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import type {
  PipelineParameterDefaultsResponse,
  PipelineParameterField,
  PipelineParameterSchema,
  PipelineParameterSchemaItem,
  PipelineSceneType
} from "@/lib/types";

type FormState = Record<string, unknown>;

export default function PipelineParametersPage() {
  const [schema, setSchema] = useState<PipelineParameterSchema | null>(null);
  const [savedDefaults, setSavedDefaults] = useState<PipelineParameterDefaultsResponse["defaults"]>({});
  const [pipeline, setPipeline] = useState<string>("litevggt_spz");
  const [sceneType, setSceneType] = useState<PipelineSceneType>("indoor");
  const [values, setValues] = useState<FormState>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setError(null);
    void Promise.all([api.pipelineParameterSchema(), api.pipelineParameterDefaults()])
      .then(([schemaData, defaultsData]) => {
        if (cancelled) return;
        setSchema(schemaData);
        setSavedDefaults(defaultsData.defaults);
        setPipeline(schemaData.pipelines[0]?.pipeline ?? "litevggt_spz");
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "参数配置加载失败");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const activePipeline = useMemo(
    () => schema?.pipelines.find((item) => item.pipeline === pipeline) ?? null,
    [pipeline, schema]
  );

  useEffect(() => {
    if (!activePipeline) return;
    setValues(effectiveDefaults(savedDefaults, activePipeline, sceneType));
    setNotice(null);
  }, [activePipeline, savedDefaults, sceneType]);

  const groups = useMemo(() => groupFields(activePipeline?.fields ?? []), [activePipeline]);

  async function save() {
    if (!activePipeline) return;
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const normalized = normalizeForSave(activePipeline.fields, values);
      const result = await api.savePipelineParameterDefaults(activePipeline.pipeline, sceneType, normalized);
      setSavedDefaults((current) => ({
        ...current,
        [activePipeline.pipeline]: {
          ...(current[activePipeline.pipeline] ?? {}),
          [sceneType]: result.options
        }
      }));
      setValues(result.options);
      setNotice("已保存当前场景默认参数");
    } catch (err) {
      setError(err instanceof Error ? err.message : "保存失败");
    } finally {
      setBusy(false);
    }
  }

  function restoreSystemDefaults() {
    if (!activePipeline) return;
    setValues({ ...(activePipeline.defaults[sceneType] ?? {}) });
    setNotice("已恢复为系统默认，保存后才会生效");
  }

  if (!schema) {
    return (
      <div className="workspace-page">
        <header className="page-header compact">
          <div>
            <p className="eyebrow">Pipeline Parameters</p>
            <h1>管线参数</h1>
          </div>
        </header>
        <div className="panel padded">{error ?? "加载中..."}</div>
      </div>
    );
  }

  return (
    <div className="workspace-page pipeline-parameters-page">
      <header className="page-header compact">
        <div>
          <p className="eyebrow">Pipeline Parameters</p>
          <h1>管线参数</h1>
        </div>
        <div className="actions">
          <button className="ghost-button" type="button" onClick={restoreSystemDefaults} disabled={busy}>
            <RotateCcw size={17} />恢复系统默认
          </button>
          <button className="button" type="button" onClick={() => void save()} disabled={busy || !activePipeline}>
            <Save size={17} />保存默认参数
          </button>
        </div>
      </header>

      <section className="pipeline-parameter-layout">
        <aside className="panel fill">
          <div className="panel-head">
            <h2>管线</h2>
            <SlidersHorizontal size={18} />
          </div>
          <div className="panel-body stack">
            <div className="pipeline-tab-list">
              {schema.pipelines.map((item) => (
                <button
                  className={`pipeline-tab ${pipeline === item.pipeline ? "active" : ""}`}
                  type="button"
                  key={item.pipeline}
                  onClick={() => setPipeline(item.pipeline)}
                >
                  <strong>{item.label}</strong>
                  <span>{item.pipeline}</span>
                </button>
              ))}
            </div>

            <div className="field">
              <label>场景</label>
              <div className="segmented">
                {schema.scene_types.map((item) => (
                  <button
                    className={sceneType === item.value ? "active" : ""}
                    type="button"
                    key={item.value}
                    onClick={() => setSceneType(item.value)}
                  >
                    {item.label}
                  </button>
                ))}
              </div>
            </div>
            {notice ? <div className="notice-box">{notice}</div> : null}
            {error ? <div className="error-box">{error}</div> : null}
          </div>
        </aside>

        <div className="panel fill">
          <div className="panel-head">
            <div>
              <h2>{activePipeline?.label}</h2>
              <p className="muted small">{sceneType === "indoor" ? "室内" : "户外"}默认参数会在新任务入队时自动套用。</p>
            </div>
            <span className="status-pill">{activePipeline?.fields.length ?? 0} 项</span>
          </div>
          <div className="panel-body scrollable stack">
            {groups.map(([group, fields]) => (
              <section className="parameter-group" key={group}>
                <div className="parameter-group-head">
                  <h3>{group}</h3>
                  <span className="muted small">{fields.length} 项</span>
                </div>
                <div className="parameter-grid">
                  {fields.map((field) => (
                    <ParameterControl
                      field={field}
                      value={values[field.key]}
                      systemValue={activePipeline?.defaults[sceneType]?.[field.key]}
                      key={field.key}
                      onChange={(value) => setValues((current) => ({ ...current, [field.key]: value }))}
                    />
                  ))}
                </div>
              </section>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}

function ParameterControl({
  field,
  value,
  systemValue,
  onChange
}: {
  field: PipelineParameterField;
  value: unknown;
  systemValue: unknown;
  onChange: (value: unknown) => void;
}) {
  return (
    <label className="parameter-control">
      <span className="parameter-label-row">
        <strong>{field.label}</strong>
        <code>{field.key}</code>
      </span>
      <span className="parameter-description">{field.description}</span>
      {renderInput(field, value, onChange)}
      <span className="parameter-default">系统默认：{formatValue(systemValue)}</span>
    </label>
  );
}

function renderInput(field: PipelineParameterField, value: unknown, onChange: (value: unknown) => void) {
  if (field.type === "boolean") {
    return (
      <div className="segmented parameter-binary">
        <button type="button" className={value === true ? "active" : ""} onClick={() => onChange(true)}>开启</button>
        <button type="button" className={value === false ? "active" : ""} onClick={() => onChange(false)}>关闭</button>
      </div>
    );
  }
  if (field.type === "deblur_switch") {
    const enabled = value !== "false" && value !== false;
    return (
      <div className="segmented parameter-binary">
        <button type="button" className={enabled ? "active" : ""} onClick={() => onChange("auto")}>开启</button>
        <button type="button" className={!enabled ? "active" : ""} onClick={() => onChange("false")}>关闭</button>
      </div>
    );
  }
  if (field.type === "select") {
    return (
      <select className="select" value={String(value ?? "")} onChange={(event) => onChange(event.target.value)}>
        {(field.options ?? []).map((option) => (
          <option value={option} key={option}>{option}</option>
        ))}
      </select>
    );
  }
  if (field.type === "nullable_number") {
    const isAuto = value == null || value === "";
    return (
      <div className="parameter-nullable">
        <input
          className="input"
          type="number"
          min={field.min}
          max={field.max}
          step={field.step ?? "any"}
          value={isAuto ? "" : String(value)}
          placeholder="默认/自动"
          onChange={(event) => onChange(event.target.value === "" ? null : Number(event.target.value))}
        />
        <button className="ghost-button" type="button" onClick={() => onChange(null)}>自动</button>
      </div>
    );
  }
  return (
    <input
      className="input"
      type="number"
      min={field.min}
      max={field.max}
      step={field.step ?? "any"}
      value={typeof value === "number" || typeof value === "string" ? String(value) : ""}
      onChange={(event) => onChange(Number(event.target.value))}
    />
  );
}

function effectiveDefaults(
  saved: PipelineParameterDefaultsResponse["defaults"],
  pipeline: PipelineParameterSchemaItem,
  sceneType: PipelineSceneType
): FormState {
  return {
    ...(pipeline.defaults[sceneType] ?? {}),
    ...((saved[pipeline.pipeline]?.[sceneType] ?? {}) as FormState)
  };
}

function groupFields(fields: PipelineParameterField[]): Array<[string, PipelineParameterField[]]> {
  const groups = new Map<string, PipelineParameterField[]>();
  for (const field of fields) {
    groups.set(field.group, [...(groups.get(field.group) ?? []), field]);
  }
  return Array.from(groups.entries());
}

function normalizeForSave(fields: PipelineParameterField[], values: FormState): FormState {
  const next: FormState = {};
  for (const field of fields) {
    const value = values[field.key];
    if (field.type === "deblur_switch") {
      next[field.key] = value === "false" || value === false ? "false" : "auto";
    } else {
      next[field.key] = value;
    }
  }
  return next;
}

function formatValue(value: unknown): string {
  if (value == null || value === "") return "自动";
  if (typeof value === "boolean") return value ? "开启" : "关闭";
  return String(value);
}
