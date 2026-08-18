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

type ResearchClaim = { text: string; evidence: string };
type CausalLink = { cause: string; effect: string; evidence: string };
type AbstractInference = { text: string; abstract_basis: string };
type ResearchStructure = {
  central_claim?: ResearchClaim;
  observation?: ResearchClaim;
  mechanism?: ResearchClaim;
  method?: ResearchClaim;
  controllable_variables?: ResearchClaim[];
  limitation_or_gap?: ResearchClaim;
  causal_links?: CausalLink[];
  boundary_conditions?: ResearchClaim[];
  inferred_assumptions?: AbstractInference[];
  alternative_explanations?: AbstractInference[];
  unknowns?: string[];
};

type AbstractDiagnostic = {
  coverage_score: number;
  grounded_elements: string[];
  missing_elements: string[];
  inferred_assumption_count: number;
  alternative_explanation_count: number;
};

type ReasoningStep = {
  kind: "abstract_evidence" | "explicit_inference" | "assumption";
  statement: string;
  anchor: string;
};

type RelatedWork = {
  title: string;
  year?: number | null;
  doi?: string;
  url?: string;
  abstract_excerpt?: string;
  databases?: string[];
};

type IdeaEvaluation = {
  testability: number;
  feasibility: number;
  potential_impact: number;
  evidence_strength: number;
  discrimination_power?: number;
  novelty_confidence: number;
  logical_support: "supported" | "partly_supported" | "weak";
  primary_concern: string;
  verdict: string;
};

type DeepIdea = {
  id: "direct_validation" | "method_transfer" | "high_risk_hypothesis";
  direction: string;
  title: string;
  hypothesis: string;
  why_it_might_work: string;
  derivation_operator?: "discriminate_cause" | "transfer_mechanism" | "invert_assumption";
  abstract_gap_targeted?: string;
  assumption_tested?: string;
  competing_explanation?: string;
  discriminating_outcome?: string;
  reasoning_chain?: ReasoningStep[];
  reasoning_steps: string[];
  minimum_test: string;
  independent_variables: string[];
  dependent_variables: string[];
  controls: string[];
  expected_result: string;
  falsification_criterion: string;
  main_risk: string;
  evidence_anchors: string[];
  novelty_search_query: string;
  related_works: RelatedWork[];
  evaluation: IdeaEvaluation;
};

type IdeaLab = {
  version?: number;
  status: "running" | "complete" | "partial" | "failed";
  ideas?: DeepIdea[];
  best_idea_id?: DeepIdea["id"];
  critic_status?: "reviewed" | "unavailable";
  overall_caveat?: string;
  novelty_disclaimer?: string;
  evidence_scope?: string;
  abstract_provenance?: string;
  abstract_diagnostic?: AbstractDiagnostic;
  idea_diversity_score?: number;
  remaining_quality_warnings?: string[];
  research_structure?: ResearchStructure;
  article_level_arxiv_context?: RelatedWork[];
  estimated_cost?: number;
  generated_at?: string;
  error?: string;
};

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
  evidence: string;
  research_structure?: ResearchStructure;
  idea_lab?: IdeaLab | null;
  idea_is_speculative: boolean;
  labels: string[];
  work_type?: string;
  publication_status?: string;
  update_status?: string;
  is_update?: boolean;
  duplicate_count?: number;
  summary_quality?: number;
  summary_source?: string;
  abstract_status?: "complete" | "excerpt" | "unavailable";
  abstract_source_url?: string;
  abstract_fetched_at?: string;
  folders?: string[];
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
    unique_count: number;
    duplicate_count: number;
    candidate_count: number;
    missing_summary_count: number;
    thin_summary_count: number;
    selected_count: number;
    excluded_count: number;
    complete_abstract_count: number;
    excerpt_abstract_count: number;
    idea_lab_count: number;
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
    browser_abstracts: boolean;
    browser_available: boolean;
    browser_last_error?: string;
    browser_verification_required: { domain: string; url: string; reason: string; affected_articles: string }[];
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
  status: { openai_configured: false, inoreader_oauth_configured: false, inoreader_connected: false, profile_configured: false, demo_mode: false, analysis_model: "gpt-5.6-luna", embedding_model: "text-embedding-3-small", data_location: "./data", local_encryption: true, browser_abstracts: true, browser_available: false, browser_verification_required: [] },
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

function matchScore(item: Recommendation) {
  return score(
    item.relevance_score * 0.65
    + item.inspiration_score * 0.25
    + item.confidence * 0.10,
  );
}

function abstractBadge(item: Recommendation) {
  if (item.abstract_status === "complete") {
    if (item.summary_source === "publisher_browser_abstract") return "FULL ABSTRACT · LIVE PUBLISHER PAGE";
    if (item.summary_source === "arxiv_feed_abstract") return "FULL ABSTRACT · ARXIV";
    if (item.summary_source === "crossref_abstract") return "FULL ABSTRACT · CROSSREF";
    if (item.summary_source === "europepmc_abstract") return "FULL ABSTRACT · EUROPE PMC";
    if (item.summary_source === "scopus_abstract") return "FULL ABSTRACT · SCOPUS";
    if (item.summary_source === "openalex_abstract") return "FULL ABSTRACT · OPENALEX";
    return "FULL ABSTRACT · PUBLISHER";
  }
  if (item.abstract_status === "excerpt") return "ABSTRACT EXCERPT ONLY";
  if (item.abstract_status === "unavailable") return "NO PUBLIC ABSTRACT";
  return "UNVERIFIED LEGACY TEXT";
}

function hasResearchClaim(claim?: ResearchClaim) {
  return Boolean(claim?.evidence && claim.text !== "Not available in the abstract.");
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
  const [ideaLoading, setIdeaLoading] = useState<string[]>([]);

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

  const openPublisherVerification = async (domain: string) => {
    try {
      const response = await fetch(`${API_BASE}/api/browser-verification`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ domain }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Could not open verification browser");
      setToast(payload.message || "Complete verification in Chrome, close it, then refresh again");
    } catch (error) {
      setToast(error instanceof Error ? error.message : "Could not open verification browser");
    }
  };

  const generateIdeaLab = async (item: Recommendation) => {
    if (!item.refresh_id) {
      setToast("This legacy article has no refresh identifier for Idea Lab generation");
      return;
    }
    const key = `${item.refresh_id}:${item.article_id}`;
    setIdeaLoading((current) => [...new Set([...current, key])]);
    try {
      const response = await fetch(
        `${API_BASE}/api/refreshes/${item.refresh_id}/articles/${encodeURIComponent(item.article_id)}/idea-lab`,
        { method: "POST" },
      );
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Could not generate the Idea Lab");
      const updateItems = (items: Recommendation[]) => items.map((candidate) =>
        candidate.refresh_id === item.refresh_id && candidate.article_id === item.article_id
          ? {
            ...candidate,
            idea_lab: payload.idea_lab,
            research_structure: payload.idea_lab?.research_structure || candidate.research_structure,
          }
          : candidate,
      );
      setData((current) => ({
        ...current,
        recommendations: updateItems(current.recommendations),
        saved: updateItems(current.saved),
        feedback_history: updateItems(current.feedback_history),
        archive: updateItems(current.archive),
      }));
      setToast(payload.idea_lab?.status === "partial" ? "Ideas ready · critic fallback is clearly marked" : "Deep Idea Lab ready");
    } catch (error) {
      setToast(error instanceof Error ? error.message : "Could not generate the Idea Lab");
    } finally {
      setIdeaLoading((current) => current.filter((itemKey) => itemKey !== key));
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
          <button className={view === "pipeline" ? "active" : ""} onClick={() => setView("pipeline")}><span>06</span> Pipeline</button>
          <button onClick={() => setPanel("settings")}><span>07</span> Settings</button>
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
            <label className="select-control"><span>TARGET</span><input aria-label="Target brief size" type="number" min="1" max="100" value={topNDraft} onChange={(event) => setTopNDraft(event.target.value)} onBlur={commitTopN} onKeyDown={(event) => { if (event.key === "Enter") event.currentTarget.blur(); }} /></label>
            <button className="refresh-button" onClick={refresh} disabled={refreshDisabled} title={data.status.inoreader_connected && !data.status.profile_configured ? "Upload a CV before refreshing" : undefined}><span className={refreshing ? "spin" : ""}>↻</span>{refreshing ? "Building brief & Idea Labs…" : "Refresh now"}</button>
          </div>
        </header>

        {view !== "pipeline" && <section className="signal-strip" aria-label="Refresh summary">
          <div><span className="signal-number">{data.run?.scanned_count || 0}</span><span className="signal-label">feed items received</span></div>
          <div><span className="signal-number">{data.run?.unique_count || data.run?.scanned_count || 0}</span><span className="signal-label">unique works</span></div>
          <div><span className="signal-number">{data.run?.candidate_count || data.run?.selected_count || 0}</span><span className="signal-label">ranked candidates</span></div>
          <div><span className="signal-number accent">{data.run?.selected_count ?? data.recommendations.length}</span><span className="signal-label">articles delivered</span></div>
          <div><span className="signal-number">{formatEstimatedCost(data.run?.estimated_cost || 0)}</span><span className="signal-label">estimated AI cost</span></div>
          <div className="last-sync"><span className="pulse-live" />Updated {relativeTime(data.run?.completed_at)}</div>
        </section>}

        {view !== "pipeline" && data.run && (data.run.unique_count > 0 || data.run.candidate_count > 0) && <div className="data-quality-note">
          <strong>Coverage:</strong> {data.run.duplicate_count || 0} duplicate feed entries merged · {data.recommendations.filter((item) => item.abstract_status === "complete").length}/{data.recommendations.length} selected articles with confirmed full abstracts · {data.run.missing_summary_count || 0} without any summary text
        </div>}

        {!!data.status.browser_verification_required.length && <section className="browser-verification-panel" aria-label="Publisher verification required">
          <div><strong>Publisher verification needed</strong><p>Complete the publisher&apos;s own check once in PaperPulse&apos;s dedicated Chrome profile, close that window, then refresh again.</p></div>
          <div className="verification-actions">{data.status.browser_verification_required.map((item) => <button key={item.domain} onClick={() => openPublisherVerification(item.domain)}>{item.domain}{item.affected_articles ? ` · ${item.affected_articles} waiting` : ""}</button>)}</div>
        </section>}

        {data.status.browser_last_error && <div className="browser-error-note"><strong>Browser abstract retrieval is unavailable:</strong> {data.status.browser_last_error}</div>}

        {view === "pipeline" ? <section className="pipeline-panel" aria-label="Refresh pipeline">
          {(() => {
            const run = data.run;
            if (!run) return <p className="pipeline-empty">No refresh has completed yet. Run a refresh to see how the pipeline narrowed your feed.</p>;
            const scanned = run.scanned_count || 0;
            const excluded = run.excluded_count || 0;
            const unique = run.unique_count || 0;
            const merged = run.duplicate_count || 0;
            const complete = run.complete_abstract_count || 0;
            const excerpt = run.excerpt_abstract_count || 0;
            const candidates = run.candidate_count || 0;
            const selected = run.selected_count ?? data.recommendations.length;
            const deep = run.idea_lab_count || 0;
            const legacy = unique > 0 && complete === 0 && excerpt === 0;
            const stages = [
              { key: "scanned", label: "Feed items received", value: scanned, note: "Unread items inside the scan window" },
              { key: "kept", label: "Research works kept", value: Math.max(scanned - excluded, 0), note: excluded ? `${excluded} corrections, news and non-scholarly items removed` : "Nothing needed removing" },
              { key: "unique", label: "Unique works", value: unique, note: merged ? `${merged} duplicate feed entries merged` : "No duplicates found" },
              { key: "complete", label: "Full abstracts resolved", value: complete, note: excerpt ? `${excerpt} left with an excerpt only` : "" },
              { key: "candidates", label: "Ranked candidates", value: candidates, note: "Scored against your research profile" },
              { key: "selected", label: "Articles delivered", value: selected, note: "Your brief" },
              { key: "deep", label: "Deep Idea Labs", value: deep, note: "Top-ranked articles with a verified abstract" },
            ].filter((stage) => stage.value > 0 || stage.key === "deep");
            const peak = Math.max(...stages.map((stage) => stage.value), 1);
            return <>
              <header className="pipeline-header">
                <div>
                  <h2>Refresh pipeline</h2>
                  <p>How {scanned.toLocaleString()} feed items became {selected} articles · {relativeTime(run.completed_at)} · {formatEstimatedCost(run.estimated_cost || 0)}</p>
                </div>
              </header>
              {legacy && <p className="pipeline-legacy">This refresh ran before abstract counts were recorded, so the abstract stage is not shown.</p>}
              <ol className="funnel">
                {stages.map((stage, index) => {
                  const previous = index > 0 ? stages[index - 1].value : stage.value;
                  const drop = previous - stage.value;
                  return <li key={stage.key} className={stage.key === "selected" ? "funnel-step accent" : "funnel-step"}>
                    <div className="funnel-head">
                      <span className="funnel-label">{stage.label}</span>
                      <span className="funnel-value">{stage.value.toLocaleString()}</span>
                    </div>
                    <div className="funnel-track"><div className="funnel-bar" style={{ width: `${Math.max((stage.value / peak) * 100, 1.5)}%` }} /></div>
                    <div className="funnel-foot">
                      {stage.note && <span>{stage.note}</span>}
                      {index > 0 && drop > 0 && <span className="funnel-drop">&minus;{drop.toLocaleString()}</span>}
                    </div>
                  </li>;
                })}
              </ol>
              {run.note && <details className="pipeline-note"><summary>Run log</summary><p>{run.note}</p></details>}
            </>;
          })()}
        </section> : <section className="content-grid">
          <div className="feed-column">
            <div className="filter-row">
              <div className="filter-tabs" role="tablist" aria-label="Recommendation label">
                {["All", "Field match", "Emerging signal", "Cross-field spark"].map((label) => <button role="tab" aria-selected={filter === label} className={filter === label ? "active" : ""} onClick={() => setFilter(label)} key={label}>{label}<span>{label === "All" ? viewItems.length : viewItems.filter((item) => item.labels.includes(label)).length}</span></button>)}
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
                const labKey = `${item.refresh_id}:${item.article_id}`;
                return (
                  <article className={`article-card ${isOpen ? "expanded" : ""}`} key={itemKey}>
                    <button className="article-summary" onClick={() => setExpanded(isOpen ? null : itemKey)} aria-expanded={isOpen}>
                      <span className="rank">{String(item.rank || index + 1).padStart(2, "0")}</span>
                      <div className="article-heading">
                        <div className="article-meta"><span>{item.source}</span><i />{item.folders?.length ? item.folders.join(", ") : item.folder}<i />{relativeTime(item.published_at)}</div>
                        <h2>{item.title}</h2>
                        <p className="recommendation-line"><span>WHY IT MATTERS</span>{item.reason}</p>
                        <div className="label-row">
                          {item.labels.map((label) => <span className={`label ${label.toLowerCase().replaceAll(" ", "-")}`} key={label}>{label}</span>)}
                          {item.work_type && <span className="label metadata-label">{item.work_type}</span>}
                          {item.update_status && item.update_status !== "New publication" && <span className={`label ${item.is_update ? "update-label" : "metadata-label"}`}>{item.update_status}</span>}
                          {(item.duplicate_count || 0) > 1 && <span className="label metadata-label">{item.duplicate_count} feed entries merged</span>}
                          <span className={`label ${item.abstract_status === "complete" ? "verified-abstract" : "low-confidence"}`}>{abstractBadge(item)}</span>
                          {item.confidence < .55 && item.abstract_status === "complete" && <span className="label low-confidence">Low abstract confidence</span>}
                        </div>
                      </div>
                      <div className="score-ring" style={{ "--score": `${matchScore(item) * 3.6}deg` } as React.CSSProperties}><strong>{matchScore(item)}</strong><small>MATCH</small></div>
                      <span className="expand-symbol">{isOpen ? "−" : "+"}</span>
                    </button>

                    {isOpen && <div className="article-detail">
                      <div className="insight-grid">
                        <div><span>CORE FINDING</span><p>{item.core_finding}</p></div>
                        <div><span>WHAT&apos;S NEW</span><p>{item.innovation}</p></div>
                        <div><span>YOUR CONNECTION</span><p>{item.connection}</p></div>
                      </div>
                      <div className="evidence-block">
                        <span>SOURCE EVIDENCE</span>
                        {item.evidence ? <blockquote>“{item.evidence}”</blockquote> : <p>{item.abstract_status === "complete" ? "No evidence quote passed validation." : "A confirmed full abstract was not available, so PaperPulse did not generate factual evidence from this item."}</p>}
                      </div>
                      <div className="score-breakdown" aria-label="Recommendation score breakdown">
                        <span><b>{score(item.relevance_score)}</b>Field match</span>
                        <span><b>{score(item.inspiration_score)}</b>Method spark</span>
                        <span><b>{score(item.novelty_score)}</b>Novelty signal</span>
                        <span><b>{score(item.confidence)}</b>Evidence confidence</span>
                      </div>
                      <ResearchStructurePanel structure={item.research_structure} />
                      <IdeaLabPanel
                        item={item}
                        loading={ideaLoading.includes(labKey)}
                        onGenerate={() => generateIdeaLab(item)}
                      />
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
                ["Emerging signal", data.recommendations.filter((item) => item.labels.includes("Emerging signal")).length, "ink"],
                ["Cross-field spark", data.recommendations.filter((item) => item.labels.includes("Cross-field spark")).length, "mint"],
              ].map(([label, count, color]) => <div className="mix-row" key={String(label)}><span><i className={String(color)} />{label}</span><strong>{count}</strong></div>)}
              <p>The selected labels can overlap; the target article count is exact when enough unique candidates exist.</p>
            </section>
          </aside>
        </section>}
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
            <div className="setting-row"><div><strong>Brief target</strong><small>PaperPulse returns exactly this many articles when enough unique eligible works exist</small></div><input className="number-setting" aria-label="Brief target" type="number" min="1" max="100" value={topNDraft} onChange={(event) => setTopNDraft(event.target.value)} onBlur={commitTopN} onKeyDown={(event) => { if (event.key === "Enter") event.currentTarget.blur(); }} /></div>
            <div className="setting-row"><div><strong>Discovery mode</strong><small>Controls how selective or exploratory each brief should be</small></div><select aria-label="Discovery mode" value={data.settings.ranking_mode} onChange={(event) => updateSettings({ ranking_mode: event.target.value as DashboardData["settings"]["ranking_mode"] })}><option value="strict">Strict</option><option value="balanced">Balanced</option><option value="exploratory">Exploratory</option></select></div>
            <div className="setting-row"><div><strong>Unread scan window</strong><small>Every refresh checks articles still unread within this period</small></div><select aria-label="Unread scan window" value={data.settings.first_sync_days} onChange={(event) => updateSettings({ first_sync_days: Number(event.target.value) })}>{[3, 7, 14, 30].map((n) => <option key={n} value={n}>{n} days</option>)}</select></div>
            <PreferenceEditor title="SOURCE RULES" names={data.source_catalog.sources} values={data.settings.source_preferences} onChange={(name, value) => updatePreference("source_preferences", name, value)} />
            <PreferenceEditor title="FOLDER RULES" names={data.source_catalog.folders} values={data.settings.folder_preferences} onChange={(name, value) => updatePreference("folder_preferences", name, value)} />
            <div className="privacy-box"><span>⌂</span><p><strong>Encrypted local library</strong>Your CV, OAuth tokens, feed cache, recommendations and feedback stay at <code>{data.status.data_location}</code>. Sensitive contents are encrypted at rest. Extracted CV text is sent to OpenAI when building the profile. During refresh, candidate text is sent to OpenAI; the top five articles also receive deep Idea Lab analysis. Technical novelty queries—not your CV—are sent to OpenAlex, Crossref and arXiv.</p></div>
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

function ResearchStructurePanel({ structure }: { structure?: ResearchStructure }) {
  if (!structure || Object.keys(structure).length === 0) return null;
  const claims: { label: string; claim?: ResearchClaim }[] = [
    { label: "Central claim", claim: structure.central_claim },
    { label: "Observation", claim: structure.observation },
    { label: "Mechanism", claim: structure.mechanism },
    { label: "Method", claim: structure.method },
    { label: "Limitation or gap", claim: structure.limitation_or_gap },
  ];
  return <section className="research-structure">
    <div className="idea-section-heading"><div><span>ABSTRACT MAP</span><h3>What the source actually supports</h3></div><small>Each populated claim requires verbatim evidence</small></div>
    <div className="structure-grid">
      {claims.map(({ label, claim }) => <div className={hasResearchClaim(claim) ? "" : "unavailable"} key={label}>
        <span>{label}</span>
        <p>{claim?.text || "Not available in the abstract."}</p>
        {hasResearchClaim(claim) && <blockquote>“{claim?.evidence}”</blockquote>}
      </div>)}
    </div>
    {!!structure.causal_links?.length && <div className="causal-map">
      <span>ABSTRACT CAUSAL CHAIN</span>
      {structure.causal_links.map((link, index) => <div key={`${link.cause}-${link.effect}-${index}`}><p><strong>{link.cause}</strong><i>→</i><strong>{link.effect}</strong></p><blockquote>“{link.evidence}”</blockquote></div>)}
    </div>}
    <div className="structure-lists">
      <div><span>CONTROLLABLE VARIABLES</span>{structure.controllable_variables?.length ? <ul>{structure.controllable_variables.map((claim, index) => <li key={`${claim.text}-${index}`}>{claim.text}</li>)}</ul> : <p>Not available in the abstract.</p>}</div>
      <div><span>STATED BOUNDARY CONDITIONS</span>{structure.boundary_conditions?.length ? <ul>{structure.boundary_conditions.map((claim, index) => <li key={`${claim.text}-${index}`}>{claim.text}</li>)}</ul> : <p>Not stated in the abstract.</p>}</div>
      <div><span>INFERRED ASSUMPTIONS</span>{structure.inferred_assumptions?.length ? <ul>{structure.inferred_assumptions.map((inference, index) => <li key={`${inference.text}-${index}`}>{inference.text}<small>Inference based on: “{inference.abstract_basis}”</small></li>)}</ul> : <p>No grounded assumptions extracted.</p>}</div>
      <div><span>ALTERNATIVE EXPLANATIONS</span>{structure.alternative_explanations?.length ? <ul>{structure.alternative_explanations.map((inference, index) => <li key={`${inference.text}-${index}`}>{inference.text}<small>Prompted by: “{inference.abstract_basis}”</small></li>)}</ul> : <p>No grounded alternative extracted.</p>}</div>
      <div><span>UNKNOWNS</span>{structure.unknowns?.length ? <ul>{structure.unknowns.map((unknown) => <li key={unknown}>{unknown}</li>)}</ul> : <p>No explicit unknowns extracted.</p>}</div>
    </div>
  </section>;
}

function ScoreMetric({ label, value }: { label: string; value: number }) {
  return <span><b>{score(value)}</b>{label}</span>;
}

function RelatedWorks({ works }: { works: RelatedWork[] }) {
  if (!works?.length) return <p className="no-prior-art">No close metadata result was retrieved. This means uncertainty, not novelty.</p>;
  return <div className="related-work-list">{works.slice(0, 5).map((work, index) => {
    const content = <><strong>{work.title}</strong><small>{[work.year, ...(work.databases || [])].filter(Boolean).join(" · ")}</small></>;
    return isHttpUrl(work.url || "")
      ? <a href={work.url} target="_blank" rel="noopener noreferrer" key={`${work.title}-${index}`}>{content}</a>
      : <div key={`${work.title}-${index}`}>{content}</div>;
  })}</div>;
}

function IdeaLabPanel({ item, loading, onGenerate }: { item: Recommendation; loading: boolean; onGenerate: () => void }) {
  const lab = item.idea_lab;
  const legacyLab = !!lab && (lab.status === "complete" || lab.status === "partial") && (lab.version || 0) < 3;
  const ready = (lab?.status === "complete" || lab?.status === "partial") && !legacyLab;
  const noFullAbstract = item.abstract_status !== "complete";
  if (!ready || !lab?.ideas?.length) return <section className="idea-lab empty-idea-lab">
    <div className="idea-section-heading"><div><span>DEEP IDEA LAB</span><h3>Three hypotheses, followed by an independent critic</h3></div><small>Abstract-grounded · prior-art assisted</small></div>
    {item.idea && <p className="legacy-idea"><strong>Legacy one-pass idea:</strong> {item.idea}</p>}
    <p>{noFullAbstract ? "PaperPulse could not confirm a complete public abstract, so it will not invent a deep analysis from an excerpt." : legacyLab ? "This Idea Lab predates the abstract-reasoning upgrade. Regenerate it to add causal alternatives, typed inference, and idea separation checks." : lab?.status === "failed" ? "The previous deep analysis did not complete. You can retry it." : "Idea Labs are generated automatically for verified-abstract articles in the first five ranks. Other verified articles are available on demand."}</p>
    <button className="generate-idea-button" onClick={onGenerate} disabled={noFullAbstract || loading || lab?.status === "running"}>{noFullAbstract ? "Full abstract required" : loading || lab?.status === "running" ? "Generating and checking prior work…" : legacyLab ? "Upgrade abstract reasoning" : lab?.status === "failed" ? "Retry deep Idea Lab" : "Generate deep Idea Lab"}</button>
  </section>;

  return <section className="idea-lab">
    <div className="idea-section-heading"><div><span>DEEP IDEA LAB</span><h3>Three testable directions</h3></div><small>{lab.evidence_scope || "Abstract only"}</small></div>
    {lab.abstract_diagnostic && <div className="abstract-diagnostic">
      <div><strong>{score(lab.abstract_diagnostic.coverage_score)}</strong><span>ABSTRACT REASONING COVERAGE</span></div>
      <p><b>Grounded:</b> {lab.abstract_diagnostic.grounded_elements.join(", ") || "none"}<br /><b>Missing:</b> {lab.abstract_diagnostic.missing_elements.join(", ") || "none"}</p>
      <small>Coverage measures usable reasoning structure in this abstract—not paper quality.{typeof lab.idea_diversity_score === "number" ? ` Idea separation: ${score(lab.idea_diversity_score)}/100.` : ""}</small>
    </div>}
    {!!lab.remaining_quality_warnings?.length && <div className="quality-warning"><strong>Remaining abstract-only cautions</strong><ul>{lab.remaining_quality_warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul></div>}
    <div className={`critic-banner ${lab.critic_status === "reviewed" ? "reviewed" : "fallback"}`}><strong>{lab.critic_status === "reviewed" ? "Independent critic completed" : "Critic fallback used"}</strong><p>{lab.overall_caveat}</p></div>
    <div className="deep-idea-list">
      {lab.ideas.map((idea) => {
        const best = lab.best_idea_id === idea.id;
        return <article className={`deep-idea-card ${best ? "best" : ""}`} key={idea.id}>
          <header><div><span>{idea.direction}{idea.derivation_operator ? ` · ${idea.derivation_operator.replaceAll("_", " ")}` : ""}</span><h4>{idea.title}</h4></div>{best && <b>CRITIC&apos;S PICK</b>}</header>
          <div className="idea-hypothesis"><span>HYPOTHESIS</span><p>{idea.hypothesis}</p></div>
          <div className="idea-score-row">
            <ScoreMetric label="Testability" value={idea.evaluation.testability} />
            <ScoreMetric label="Feasibility" value={idea.evaluation.feasibility} />
            <ScoreMetric label="Impact" value={idea.evaluation.potential_impact} />
            <ScoreMetric label="Evidence" value={idea.evaluation.evidence_strength} />
            {typeof idea.evaluation.discrimination_power === "number" && <ScoreMetric label="Discrimination" value={idea.evaluation.discrimination_power} />}
            <ScoreMetric label="Novelty confidence" value={idea.evaluation.novelty_confidence} />
          </div>
          <div className="idea-detail-grid">
            <div><span>WHY IT MIGHT WORK</span><p>{idea.why_it_might_work}</p></div>
            {idea.abstract_gap_targeted && <div><span>ABSTRACT GAP TARGETED</span><p>{idea.abstract_gap_targeted}</p></div>}
            {idea.assumption_tested && <div><span>ASSUMPTION UNDER TEST</span><p>{idea.assumption_tested}</p></div>}
            {idea.competing_explanation && <div><span>STRONGEST ALTERNATIVE</span><p>{idea.competing_explanation}</p></div>}
            {idea.discriminating_outcome && <div className="discriminator"><span>DISCRIMINATING OUTCOME</span><p>{idea.discriminating_outcome}</p></div>}
            <div><span>MINIMUM DECISIVE TEST</span><p>{idea.minimum_test}</p></div>
            <div><span>EXPECTED RESULT</span><p>{idea.expected_result}</p></div>
            <div className="falsifier"><span>WHAT WOULD FALSIFY IT</span><p>{idea.falsification_criterion}</p></div>
            <div><span>MAIN RISK</span><p>{idea.main_risk}</p></div>
            <div><span>CRITIC VERDICT</span><p>{idea.evaluation.verdict}</p><small>{idea.evaluation.primary_concern}</small></div>
          </div>
          <div className="experiment-map">
            <div><span>INDEPENDENT VARIABLES</span><ul>{idea.independent_variables.map((value) => <li key={value}>{value}</li>)}</ul></div>
            <div><span>DEPENDENT VARIABLES</span><ul>{idea.dependent_variables.map((value) => <li key={value}>{value}</li>)}</ul></div>
            <div><span>CONTROLS</span><ul>{idea.controls.map((value) => <li key={value}>{value}</li>)}</ul></div>
          </div>
          {!!idea.reasoning_chain?.length ? <div className="reasoning-chain"><span>TYPED REASONING CHAIN</span><ol>{idea.reasoning_chain.map((step, index) => <li key={`${step.statement}-${index}`}><i className={`reasoning-kind ${step.kind}`}>{step.kind.replaceAll("_", " ")}</i>{step.statement}{step.anchor !== "none" && <small>anchor: {step.anchor}</small>}</li>)}</ol></div> : !!idea.reasoning_steps?.length && <div className="reasoning-chain"><span>REASONING CHAIN</span><ol>{idea.reasoning_steps.map((step, index) => <li key={`${step}-${index}`}>{step}</li>)}</ol></div>}
          <div className="evidence-anchor-row"><span>EVIDENCE ANCHORS</span>{idea.evidence_anchors.map((anchor) => <i key={anchor}>{anchor}</i>)}</div>
          <div className="prior-art"><span>NEAREST RETRIEVED WORK</span><RelatedWorks works={idea.related_works} /></div>
        </article>;
      })}
    </div>
    {lab.article_level_arxiv_context && lab.article_level_arxiv_context.length > 0 && <div className="arxiv-context"><span>ADDITIONAL ARXIV CONTEXT</span><RelatedWorks works={lab.article_level_arxiv_context} /></div>}
    <p className="novelty-disclaimer">{lab.novelty_disclaimer}</p>
  </section>;
}
