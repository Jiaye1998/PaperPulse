"use client";

import { ChangeEvent, useEffect, useMemo, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

type Feedback =
  | "relevant"
  | "inspiring"
  | "not_useful"
  | "save_for_later"
  | "already_known"
  | "read";

type Recommendation = {
  refresh_id?: number;
  run_completed_at?: string;
  article_id: string;
  rank: number;
  title: string;
  source: string;
  folder: string;
  published_at: string;
  url: string;
  relevance_score: number;
  novelty_score: number;
  inspiration_score: number;
  confidence: number;
  reason: string;
  core_finding: string;
  innovation: string;
  connection: string;
  idea: string;
  idea_is_speculative: boolean;
  labels: string[];
  feedback?: Feedback | null;
};

type ResearchProfile = {
  name: string;
  headline: string;
  domains: string[];
  methods: string[];
  systems: string[];
  current_questions: string[];
  adjacent_fields: string[];
  keywords: string[];
};

type DashboardData = {
  run: {
    scanned_count: number;
    selected_count: number;
    estimated_cost: number;
    completed_at: string;
    note: string;
    status: string;
  } | null;
  recommendations: Recommendation[];
  saved: Recommendation[];
  feedback_history: Recommendation[];
  archive: Recommendation[];
  history_runs: { id: number; completed_at: string; scanned_count: number; selected_count: number; status: string }[];
  source_catalog: { sources: string[]; folders: string[] };
  feedback_counts: Record<string, number>;
  profile: { filename: string; profile: ResearchProfile; updated_at: string } | null;
  settings: {
    top_n: number;
    first_sync_days: number;
    candidate_multiplier: number;
    ranking_mode: "strict" | "balanced" | "exploratory";
    source_preferences: Record<string, "boost" | "normal" | "lower" | "exclude">;
    folder_preferences: Record<string, "boost" | "normal" | "lower" | "exclude">;
  };
  status: {
    openai_configured: boolean;
    inoreader_oauth_configured: boolean;
    inoreader_connected: boolean;
    inoreader_last_error?: string;
    profile_configured: boolean;
    demo_mode: boolean;
    analysis_model: string;
    embedding_model: string;
    data_location: string;
    local_encryption: boolean;
  };
};

const fallbackProfile: ResearchProfile = {
  name: "Researcher",
  headline: "Upload a CV to configure your research interests",
  domains: [],
  methods: [],
  systems: [],
  current_questions: [],
  adjacent_fields: [],
  keywords: [],
};

const fallbackData: DashboardData = {
  run: null,
  recommendations: [],
  saved: [],
  feedback_history: [],
  archive: [],
  history_runs: [],
  source_catalog: { sources: [], folders: [] },
  feedback_counts: {},
  profile: null,
  settings: { top_n: 20, first_sync_days: 7, candidate_multiplier: 2, ranking_mode: "balanced", source_preferences: {}, folder_preferences: {} },
  status: { openai_configured: false, inoreader_oauth_configured: false, inoreader_connected: false, profile_configured: false, demo_mode: false, analysis_model: "gpt-5.6-luna", embedding_model: "text-embedding-3-small", data_location: "./data", local_encryption: true },
};

function normalizeData(payload: Partial<DashboardData>): DashboardData {
  return {
    ...fallbackData,
    ...payload,
    recommendations: payload.recommendations || [],
    saved: payload.saved || [],
    feedback_history: payload.feedback_history || [],
    archive: payload.archive || [],
    history_runs: payload.history_runs || [],
    source_catalog: payload.source_catalog || { sources: [], folders: [] },
    feedback_counts: payload.feedback_counts || {},
    settings: {
      ...fallbackData.settings,
      ...(payload.settings || {}),
      source_preferences: payload.settings?.source_preferences || {},
      folder_preferences: payload.settings?.folder_preferences || {},
    },
    status: { ...fallbackData.status, ...(payload.status || {}) },
  };
}

const feedbackOptions: { value: Feedback; label: string; symbol: string }[] = [
  { value: "relevant", label: "Relevant", symbol: "◎" },
  { value: "inspiring", label: "Inspiring", symbol: "✦" },
  { value: "not_useful", label: "Not useful", symbol: "–" },
  { value: "save_for_later", label: "Save", symbol: "◇" },
  { value: "already_known", label: "Known", symbol: "✓" },
  { value: "read", label: "Read", symbol: "○" },
];

function score(value: number) {
  return Math.round(value * 100);
}

function isHttpUrl(value: string) {
  try {
    const url = new URL(value);
    return url.protocol === "http:" || url.protocol === "https:";
  } catch {
    return false;
  }
}

function formatEstimatedCost(value: number) {
  if (value <= 0) return "$0.00";
  return value < 0.01 ? `$${value.toFixed(4)}` : `$${value.toFixed(2)}`;
}

function relativeTime(value?: string) {
  if (!value) return "Not refreshed yet";
  const hours = Math.max(0, Math.floor((Date.now() - new Date(value).getTime()) / 3600000));
  if (hours < 1) return "Just now";
  if (hours === 1) return "1 hour ago";
  if (hours < 24) return `${hours} hours ago`;
  return `${Math.floor(hours / 24)} days ago`;
}

export default function PaperPulse() {
  const [data, setData] = useState<DashboardData>(fallbackData);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [view, setView] = useState("today");
  const [filter, setFilter] = useState("All");
  const [expanded, setExpanded] = useState<string | null>(null);
  const [panel, setPanel] = useState<"profile" | "settings" | null>(null);
  const [toast, setToast] = useState("");
  const [profileDraft, setProfileDraft] = useState<ResearchProfile>(fallbackProfile);
  const [uploading, setUploading] = useState(false);
  const [topNDraft, setTopNDraft] = useState("20");
  const [archiveQuery, setArchiveQuery] = useState("");
  const [archiveRun, setArchiveRun] = useState("all");

  useEffect(() => {
    fetch(`${API_BASE}/api/dashboard`)
      .then((response) => {
        if (!response.ok) throw new Error("Local API unavailable");
        return response.json();
      })
      .then((payload: DashboardData) => {
        const normalized = normalizeData(payload);
        setData(normalized);
        setTopNDraft(String(normalized.settings.top_n));
        if (normalized.profile) setProfileDraft(normalized.profile.profile);
      })
      .catch(() => setToast("Local API unavailable · no articles are being shown"))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const outcome = params.get("inoreader");
    if (!outcome) return;

    const message = params.get("message");
    window.history.replaceState({}, "", window.location.pathname);

    if (outcome === "error") {
      const timer = window.setTimeout(
        () => setToast(message ? `Inoreader connection failed: ${message}` : "Inoreader connection failed"),
        0,
      );
      return () => window.clearTimeout(timer);
    }

    let cancelled = false;
    fetch(`${API_BASE}/api/dashboard`)
      .then((response) => {
        if (!response.ok) throw new Error("Could not verify the Inoreader connection");
        return response.json();
      })
      .then((payload: DashboardData) => {
        if (cancelled) return;
        const normalized = normalizeData(payload);
        setData(normalized);
        if (normalized.profile) setProfileDraft(normalized.profile.profile);
        setToast(
          normalized.status.inoreader_connected
            ? "Inoreader connected — your feed is ready"
            : "Inoreader authorization returned, but no connection token was saved",
        );
      })
      .catch((error) => {
        if (!cancelled) setToast(error instanceof Error ? error.message : "Could not verify the Inoreader connection");
      });

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(""), 4200);
    return () => window.clearTimeout(timer);
  }, [toast]);

  const viewItems = useMemo(
    () => view === "saved" ? data.saved : view === "learning" ? data.feedback_history : view === "archive" ? data.archive : data.recommendations,
    [data.archive, data.feedback_history, data.recommendations, data.saved, view],
  );

  const visible = useMemo(
    () => viewItems.filter((item) => {
      if (filter !== "All" && !item.labels.includes(filter)) return false;
      if (view !== "archive") return true;
      if (archiveRun !== "all" && item.refresh_id !== Number(archiveRun)) return false;
      const haystack = [item.title, item.source, item.folder, item.reason, item.core_finding, item.innovation, item.connection, item.idea].join(" ").toLowerCase();
      return haystack.includes(archiveQuery.trim().toLowerCase());
    }),
    [archiveQuery, archiveRun, filter, view, viewItems],
  );

  useEffect(() => {
    if (!panel) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setPanel(null);
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [panel]);

  const refresh = async () => {
    setRefreshing(true);
    try {
      const response = await fetch(`${API_BASE}/api/refresh`, { method: "POST" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Refresh failed");
      setData(normalizeData(payload));
      setToast(payload.run?.note || "Your brief is ready");
    } catch (error) {
      setToast(error instanceof Error ? error.message : "Could not refresh");
    } finally {
      setRefreshing(false);
    }
  };

  const updateSettings = async (patch: Partial<DashboardData["settings"]>) => {
    const previous = data.settings;
    setData((current) => ({ ...current, settings: { ...current.settings, ...patch } }));
    try {
      const response = await fetch(`${API_BASE}/api/settings`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patch),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Could not save settings");
      setData((current) => ({ ...current, settings: payload }));
    } catch (error) {
      setData((current) => ({ ...current, settings: previous }));
      setToast(error instanceof Error ? error.message : "Could not save settings");
    }
  };

  const commitTopN = () => {
    const parsed = Number(topNDraft);
    if (!Number.isInteger(parsed) || parsed < 1 || parsed > 100) {
      setTopNDraft(String(data.settings.top_n));
      setToast("Shortlist size must be a whole number from 1 to 100");
      return;
    }
    if (parsed !== data.settings.top_n) updateSettings({ top_n: parsed });
  };

  const updatePreference = (kind: "source_preferences" | "folder_preferences", name: string, value: "boost" | "normal" | "lower" | "exclude") => {
    updateSettings({ [kind]: { ...data.settings[kind], [name]: value } });
  };

  const giveFeedback = async (articleId: string, value: Feedback) => {
    const previous = data;
    setData((current) => ({
      ...current,
      recommendations: current.recommendations.map((item) =>
        item.article_id === articleId ? { ...item, feedback: value } : item,
      ),
    }));
    try {
      const response = await fetch(`${API_BASE}/api/articles/${encodeURIComponent(articleId)}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ value }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Could not save feedback");
      setData(normalizeData(payload));
      setToast("Feedback saved · future rankings will adapt");
    } catch (error) {
      setData(previous);
      setToast(error instanceof Error ? error.message : "Could not save feedback");
    }
  };

  const uploadCv = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setUploading(true);
    const body = new FormData();
    body.append("file", file);
    try {
      const response = await fetch(`${API_BASE}/api/profile/cv`, { method: "POST", body });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "CV analysis failed");
      setProfileDraft(payload.profile);
      setData((current) => ({ ...current, profile: { filename: file.name, profile: payload.profile, updated_at: new Date().toISOString() }, status: { ...current.status, profile_configured: true } }));
      setToast("Research profile extracted — review and save it");
    } catch (error) {
      setToast(error instanceof Error ? error.message : "CV upload failed");
    } finally {
      setUploading(false);
      event.target.value = "";
    }
  };

  const saveProfile = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/profile`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ profile: profileDraft }),
      });
      if (!response.ok) throw new Error("Could not save the profile");
      setData((current) => ({ ...current, profile: { filename: current.profile?.filename || "manual-profile", profile: profileDraft, updated_at: new Date().toISOString() }, status: { ...current.status, profile_configured: true } }));
      setPanel(null);
      setToast("Research profile saved");
    } catch (error) {
      setToast(error instanceof Error ? error.message : "Could not save the profile");
    }
  };

  const connectInoreader = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/inoreader/auth/start`);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Connection setup is incomplete");
      window.location.href = payload.authorization_url;
    } catch (error) {
      setToast(error instanceof Error ? error.message : "Could not connect Inoreader");
    }
  };

  const profile = data.profile?.profile || fallbackProfile;
  const positiveFeedback = ["relevant", "inspiring", "save_for_later"].reduce(
    (total, key) => total + (data.feedback_counts[key] || 0),
    0,
  );
  const refreshDisabled = refreshing || (data.status.inoreader_connected && !data.status.profile_configured) || (!data.status.inoreader_connected && !data.status.demo_mode);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand" aria-label="PaperPulse home">
          <span className="brand-mark"><i /><i /><i /></span>
          <span>PaperPulse</span>
        </div>
        <p className="brand-tagline">Research intelligence,<br />tuned to you.</p>

        <nav className="nav-list" aria-label="Primary navigation">
          <button className={view === "today" ? "active" : ""} onClick={() => setView("today")}><span>01</span> Today&apos;s brief</button>
          <button className={view === "saved" ? "active" : ""} onClick={() => setView("saved")}><span>02</span> Saved <b>{data.saved.length}</b></button>
          <button className={view === "archive" ? "active" : ""} onClick={() => setView("archive")}><span>03</span> Brief archive <b>{data.archive.length}</b></button>
          <button onClick={() => setPanel("profile")}><span>04</span> Research profile</button>
          <button className={view === "learning" ? "active" : ""} onClick={() => setView("learning")}><span>05</span> Feedback learning</button>
          <button onClick={() => setPanel("settings")}><span>06</span> Settings</button>
        </nav>

        <div className="sidebar-bottom">
          <div className="connection-card">
            <div className="eyebrow">FEED CONNECTION</div>
            <div className="connection-row">
              <span className={`status-dot ${data.status.inoreader_connected ? "online" : ""}`} />
              <div><strong>Inoreader</strong><small>{data.status.inoreader_connected ? "Connected · read only" : "Not connected"}</small></div>
            </div>
            {!data.status.inoreader_connected && <button onClick={() => setPanel("settings")}>Set up connection →</button>}
          </div>
          <div className="privacy-note"><span>⌂</span><p><strong>Your library stays local</strong><br />CV, history and feedback live on this device.</p></div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <div className="eyebrow">{new Intl.DateTimeFormat("en", { weekday: "long", month: "long", day: "numeric" }).format(new Date()).toUpperCase()}</div>
            <h1>Your research pulse.</h1>
            <p>{loading ? "Opening your local library…" : `A focused brief shaped by ${profile.name === "Researcher" ? "your CV" : profile.name.split(" ")[0] + "’s research lens"}.`}</p>
          </div>
          <div className="top-actions">
            <label className="select-control"><span>TOP</span><input aria-label="Maximum shortlist size" type="number" min="1" max="100" value={topNDraft} onChange={(event) => setTopNDraft(event.target.value)} onBlur={commitTopN} onKeyDown={(event) => { if (event.key === "Enter") event.currentTarget.blur(); }} /></label>
            <button className="refresh-button" onClick={refresh} disabled={refreshDisabled} title={data.status.inoreader_connected && !data.status.profile_configured ? "Upload a CV before refreshing" : undefined}><span className={refreshing ? "spin" : ""}>↻</span>{refreshing ? "Building brief…" : "Refresh now"}</button>
          </div>
        </header>

        <section className="signal-strip" aria-label="Refresh summary">
          <div><span className="signal-number">{data.run?.scanned_count || 0}</span><span className="signal-label">unread scanned</span></div>
          <div><span className="signal-number accent">{data.run?.selected_count ?? data.recommendations.length}</span><span className="signal-label">signals selected</span></div>
          <div><span className="signal-number">{new Set(data.recommendations.map((item) => item.source)).size}</span><span className="signal-label">sources surfaced</span></div>
          <div><span className="signal-number">{formatEstimatedCost(data.run?.estimated_cost || 0)}</span><span className="signal-label">estimated AI cost</span></div>
          <div className="last-sync"><span className="pulse-live" />Updated {relativeTime(data.run?.completed_at)}</div>
        </section>

        <section className="content-grid">
          <div className="feed-column">
            <div className="filter-row">
              <div className="filter-tabs" role="tablist" aria-label="Recommendation label">
                {["All", "Field match", "Frontier", "Cross-field spark"].map((label) => <button role="tab" aria-selected={filter === label} className={filter === label ? "active" : ""} onClick={() => setFilter(label)} key={label}>{label}<span>{label === "All" ? viewItems.length : viewItems.filter((item) => item.labels.includes(label)).length}</span></button>)}
              </div>
              <span className="sort-note">Ranked for your profile</span>
            </div>

            {view === "archive" && <div className="archive-tools">
              <label><span>SEARCH BRIEFS</span><input type="search" placeholder="Title, source, idea, connection…" value={archiveQuery} onChange={(event) => setArchiveQuery(event.target.value)} /></label>
              <label><span>REFRESH</span><select value={archiveRun} onChange={(event) => setArchiveRun(event.target.value)}><option value="all">All refreshes</option>{data.history_runs.map((run) => <option value={run.id} key={run.id}>{new Date(run.completed_at).toLocaleDateString()} · {run.selected_count} selected</option>)}</select></label>
            </div>}

            <div className="article-list">
              {visible.length === 0 && <div className="empty-state"><span>◇</span><h2>No signals here yet.</h2><p>{view === "today" ? "Refresh to scan the unread items currently available in Inoreader." : view === "saved" ? "Use Save on an article and it will remain here across future refreshes." : view === "archive" ? "No historical brief matches this search or refresh." : "Give feedback on articles and your history will appear here."}</p></div>}
              {visible.map((item, index) => {
                const itemKey = `${view}-${item.refresh_id || "current"}-${item.article_id}`;
                const isOpen = expanded === itemKey;
                return (
                  <article className={`article-card ${isOpen ? "expanded" : ""}`} key={itemKey}>
                    <button className="article-summary" onClick={() => setExpanded(isOpen ? null : itemKey)} aria-expanded={isOpen}>
                      <span className="rank">{String(item.rank || index + 1).padStart(2, "0")}</span>
                      <div className="article-heading">
                        <div className="article-meta"><span>{item.source}</span><i />{item.folder}<i />{relativeTime(item.published_at)}</div>
                        <h2>{item.title}</h2>
                        <p className="recommendation-line"><span>WHY IT MATTERS</span>{item.reason}</p>
                        <div className="label-row">{item.labels.map((label) => <span className={`label ${label.toLowerCase().replaceAll(" ", "-")}`} key={label}>{label}</span>)}{item.confidence < .55 && <span className="label low-confidence">Thin summary</span>}</div>
                      </div>
                      <div className="score-ring" style={{ "--score": `${score(Math.max(item.relevance_score, item.inspiration_score)) * 3.6}deg` } as React.CSSProperties}><strong>{score(Math.max(item.relevance_score, item.inspiration_score))}</strong><small>FIT</small></div>
                      <span className="expand-symbol">{isOpen ? "−" : "+"}</span>
                    </button>

                    {isOpen && <div className="article-detail">
                      <div className="insight-grid">
                        <div><span>CORE FINDING</span><p>{item.core_finding}</p></div>
                        <div><span>WHAT&apos;S NEW</span><p>{item.innovation}</p></div>
                        <div><span>YOUR CONNECTION</span><p>{item.connection}</p></div>
                        <div className="idea-block"><span>IDEA SPARK <b>{item.idea_is_speculative ? "SPECULATIVE" : "GROUNDED"}</b></span><p>{item.idea}</p></div>
                      </div>
                      <div className="article-footer">
                        {isHttpUrl(item.url) ? <a href={item.url} target="_blank" rel="noopener noreferrer">Open original <span>↗</span></a> : <span className="source-unavailable">Original link unavailable</span>}
                        <div className="feedback-group"><small>FEEDBACK &amp; LIBRARY</small>{feedbackOptions.map((option) => <button title={option.label} aria-label={option.label} className={item.feedback === option.value ? "selected" : ""} onClick={() => giveFeedback(item.article_id, option.value)} key={option.value}><span>{option.symbol}</span>{option.label}</button>)}</div>
                      </div>
                    </div>}
                  </article>
                );
              })}
            </div>
          </div>

          <aside className="insight-rail">
            <section className="lens-card">
              <div className="section-title"><span>YOUR RESEARCH LENS</span><button onClick={() => setPanel("profile")}>Edit</button></div>
              <h3>{profile.headline}</h3>
              <div className="topic-cloud">{profile.domains.slice(0, 5).map((topic, index) => <span className={index < 2 ? "primary" : ""} key={topic}>{topic}</span>)}</div>
              <div className="lens-footer"><span>Based on</span><strong>{data.profile?.filename === "demo-profile" ? "Demo profile" : data.profile?.filename || "No CV yet"}</strong></div>
            </section>

            <section className="learning-card">
              <div className="section-title"><span>LEARNING LOOP</span><i className="learning-pulse" /></div>
              <div className="learning-score"><strong>{positiveFeedback}</strong><span>positive<br />signals</span></div>
              <p>Relevant, Inspiring, Not useful and Known tune future rankings. Save is positive; Read is organizational only.</p>
              <div className="learning-bar"><i style={{ width: `${Math.min(100, 18 + positiveFeedback * 12)}%` }} /></div>
              <small>{positiveFeedback < 5 ? `${5 - positiveFeedback} more signals to sharpen your lens` : "Your lens is adapting"}</small>
            </section>

            <section className="method-card">
              <span className="eyebrow">TODAY&apos;S MIX</span>
              {[
                ["Field match", data.recommendations.filter((item) => item.labels.includes("Field match")).length, "coral"],
                ["Frontier", data.recommendations.filter((item) => item.labels.includes("Frontier")).length, "ink"],
                ["Cross-field spark", data.recommendations.filter((item) => item.labels.includes("Cross-field spark")).length, "mint"],
              ].map(([label, count, color]) => <div className="mix-row" key={String(label)}><span><i className={String(color)} />{label}</span><strong>{count}</strong></div>)}
              <p>No fixed quota. PaperPulse follows the strongest signals in each batch.</p>
            </section>
          </aside>
        </section>
      </main>

      {panel && <div className="modal-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && setPanel(null)}>
        <section className="drawer" role="dialog" aria-modal="true" aria-label={panel === "profile" ? "Research profile" : "Settings"}>
          <button className="drawer-close" onClick={() => setPanel(null)} aria-label="Close">×</button>
          {panel === "profile" ? <>
            <div className="drawer-kicker">PERSONALIZATION</div><h2>Your research profile</h2><p className="drawer-intro">PaperPulse extracts this lens from your CV. Review it—the best recommendations start with an accurate profile.</p>
            <label className="upload-zone"><input type="file" accept=".pdf,.docx" onChange={uploadCv} disabled={uploading} /><span>{uploading ? "Analyzing your CV…" : "Upload a new CV"}</span><small>PDF or DOCX · stored only in your local data folder</small></label>
            <label className="field-label">NAME<input value={profileDraft.name} onChange={(event) => setProfileDraft({ ...profileDraft, name: event.target.value })} /></label>
            <label className="field-label">PROFILE HEADLINE<input value={profileDraft.headline} onChange={(event) => setProfileDraft({ ...profileDraft, headline: event.target.value })} /></label>
            <ListField label="CORE DOMAINS" value={profileDraft.domains} onChange={(domains) => setProfileDraft({ ...profileDraft, domains })} />
            <ListField label="METHODS" value={profileDraft.methods} onChange={(methods) => setProfileDraft({ ...profileDraft, methods })} />
            <ListField label="SYSTEMS & MATERIALS" value={profileDraft.systems} onChange={(systems) => setProfileDraft({ ...profileDraft, systems })} />
            <ListField label="CURRENT RESEARCH QUESTIONS" value={profileDraft.current_questions} onChange={(current_questions) => setProfileDraft({ ...profileDraft, current_questions })} />
            <ListField label="ADJACENT FIELDS FOR INSPIRATION" value={profileDraft.adjacent_fields} onChange={(adjacent_fields) => setProfileDraft({ ...profileDraft, adjacent_fields })} />
            <ListField label="KEYWORDS" value={profileDraft.keywords} onChange={(keywords) => setProfileDraft({ ...profileDraft, keywords })} />
            <button className="primary-wide" onClick={saveProfile}>Save research lens</button>
          </> : <>
            <div className="drawer-kicker">LOCAL SETUP</div><h2>Connections & ranking</h2><p className="drawer-intro">Keys stay in your local <code>.env</code>. PaperPulse requests read-only access and never marks items read.</p>
            <div className="setup-card"><div><span className={`status-dot ${data.status.openai_configured ? "online" : ""}`} /><strong>OpenAI</strong><small>{data.status.openai_configured ? `${data.status.analysis_model} ready` : "Add OPENAI_API_KEY to .env"}</small></div><span>{data.status.openai_configured ? "Ready" : "Required"}</span></div>
            <div className="setup-card"><div><span className={`status-dot ${data.status.inoreader_connected ? "online" : ""}`} /><strong>Inoreader</strong><small>{data.status.inoreader_connected ? "Connected with read-only OAuth" : data.status.inoreader_last_error ? `Last connection error: ${data.status.inoreader_last_error}` : data.status.inoreader_oauth_configured ? "OAuth app ready to connect" : "Add OAuth credentials to .env"}</small></div><button onClick={connectInoreader}>{data.status.inoreader_connected ? "Reconnect" : "Connect"}</button></div>
            <div className="setting-row"><div><strong>Shortlist size</strong><small>Any whole number from 1 to 100; this is a maximum, not a quota</small></div><input className="number-setting" aria-label="Shortlist size" type="number" min="1" max="100" value={topNDraft} onChange={(event) => setTopNDraft(event.target.value)} onBlur={commitTopN} onKeyDown={(event) => { if (event.key === "Enter") event.currentTarget.blur(); }} /></div>
            <div className="setting-row"><div><strong>Discovery mode</strong><small>Controls how selective or exploratory each brief should be</small></div><select aria-label="Discovery mode" value={data.settings.ranking_mode} onChange={(event) => updateSettings({ ranking_mode: event.target.value as DashboardData["settings"]["ranking_mode"] })}><option value="strict">Strict</option><option value="balanced">Balanced</option><option value="exploratory">Exploratory</option></select></div>
            <div className="setting-row"><div><strong>Unread scan window</strong><small>Every refresh checks articles still unread within this period</small></div><select aria-label="Unread scan window" value={data.settings.first_sync_days} onChange={(event) => updateSettings({ first_sync_days: Number(event.target.value) })}>{[3, 7, 14, 30].map((n) => <option key={n} value={n}>{n} days</option>)}</select></div>
            <PreferenceEditor title="SOURCE RULES" names={data.source_catalog.sources} values={data.settings.source_preferences} onChange={(name, value) => updatePreference("source_preferences", name, value)} />
            <PreferenceEditor title="FOLDER RULES" names={data.source_catalog.folders} values={data.settings.folder_preferences} onChange={(name, value) => updatePreference("folder_preferences", name, value)} />
            <div className="privacy-box"><span>⌂</span><p><strong>Encrypted local library</strong>Your CV, OAuth tokens, feed cache, recommendations and feedback stay at <code>{data.status.data_location}</code>. Sensitive contents are encrypted at rest. Extracted CV text is sent to OpenAI when building the profile. During ranking, candidate titles and summaries are sent for embeddings, and shortlisted candidates are sent for detailed analysis.</p></div>
          </>}
        </section>
      </div>}

      {toast && <div className="toast"><span>●</span>{toast}</div>}
    </div>
  );
}

function ListField({ label, value, onChange }: { label: string; value: string[]; onChange: (value: string[]) => void }) {
  return <label className="field-label">{label}<textarea rows={2} value={value.join(", ")} onChange={(event) => onChange(event.target.value.split(",").map((item) => item.trim()).filter(Boolean))} /></label>;
}

function PreferenceEditor({ title, names, values, onChange }: { title: string; names: string[]; values: Record<string, "boost" | "normal" | "lower" | "exclude">; onChange: (name: string, value: "boost" | "normal" | "lower" | "exclude") => void }) {
  if (!names.length) return <section className="preference-section"><span className="field-label">{title}</span><p>No cached sources yet. Connect Inoreader and refresh once.</p></section>;
  return <section className="preference-section"><span className="field-label">{title}</span><div className="preference-list">{names.map((name) => <label key={name}><span title={name}>{name}</span><select aria-label={`${name} preference`} value={values[name] || "normal"} onChange={(event) => onChange(name, event.target.value as "boost" | "normal" | "lower" | "exclude")}><option value="boost">Boost</option><option value="normal">Normal</option><option value="lower">Lower</option><option value="exclude">Exclude</option></select></label>)}</div></section>;
}
