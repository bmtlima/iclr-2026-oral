import papersDoc from "../../data/papers.json";
import topicsDoc from "../../data/topics.json";

export type MatchMethod = "exact" | "fuzzy" | "unmatched" | "ambiguous";

export interface Paper {
  id: string;
  forum: string;
  openreview_url: string | null;
  pdf_url: string | null;
  title: string;
  authors: string[];
  authorids: string[];
  abstract: string;
  tldr: string | null;
  keywords: string[];
  primary_area: string | null;
  session_date: string | null;
  session_start: string | null;
  session_end: string | null;
  room: string | null;
  iclrcc_url: string | null;
  match_confidence: number | null;
  match_method: MatchMethod;
  topic_slug: string;
  topic_confidence: "auto" | "manual";
}

export interface Topic {
  slug: string;
  label: string;
  description: string;
  keywords: string[];
}

export interface Enrichment {
  id: string;
  one_sentence_summary: string;
  contributions: string[];
  methods_used: string[];
  datasets_used: string[];
  limitations: string[];
  future_work: string[];
  source_sections_found: {
    conclusion: boolean;
    limitations: boolean;
    future_work: boolean;
    discussion: boolean;
  };
  pdf_extraction_status: "ok" | "partial" | "failed";
  enriched_at: string;
  input_char_count: number;
  model_stop_reason: string;
  cost_usd_estimate: number;
}

interface EnrichedDoc {
  generated_at: string;
  model: string;
  schema_version: number;
  enriched: Record<string, Enrichment>;
}

export const papers = (papersDoc as { papers: Paper[] }).papers;
export const topics: Topic[] = (topicsDoc as { topics: Topic[] }).topics;
export const topicBySlug: Map<string, Topic> = new Map(topics.map((t) => [t.slug, t]));

import enrichedDoc from "../../data/enriched.json";
const _enrichedDoc = enrichedDoc as EnrichedDoc;
export const enriched = _enrichedDoc.enriched || {};
export const enrichedModel = _enrichedDoc.model || "";

interface TrendTheme {
  slug: string;
  headline: string;
  explanation: string;
  paper_ids: string[];
  representative_quotes: { paper_id: string; quote: string }[];
}
interface TrendsDoc {
  generated_at: string;
  model: string;
  themes: TrendTheme[];
}
import trendsDoc from "../../data/trends.json";
const _trendsDoc = trendsDoc as TrendsDoc;
export const trends: TrendsDoc | null = _trendsDoc.themes.length > 0 ? _trendsDoc : null;

export function papersByTopic(slug: string): Paper[] {
  return papers
    .filter((p) => p.topic_slug === slug)
    .sort((a, b) => a.title.localeCompare(b.title));
}

export function topicCount(slug: string): number {
  return papers.reduce((n, p) => n + (p.topic_slug === slug ? 1 : 0), 0);
}

export function getPaper(id: string): Paper | undefined {
  return papers.find((p) => p.id === id);
}

export function formatSessionDate(p: Paper): string | null {
  if (!p.session_date) return null;
  // e.g. "2026-04-23" -> "Thu, Apr 23"
  const [y, m, d] = p.session_date.split("-").map(Number);
  const dt = new Date(Date.UTC(y, m - 1, d));
  const wd = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][dt.getUTCDay()];
  const mon = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][dt.getUTCMonth()];
  return `${wd}, ${mon} ${dt.getUTCDate()}`;
}

export function formatSessionTime(p: Paper): string | null {
  if (!p.session_start) return null;
  const fmt = (t: string) => {
    const [h, m] = t.split(":").map(Number);
    const mer = h >= 12 ? "PM" : "AM";
    const hh = h === 0 ? 12 : h > 12 ? h - 12 : h;
    return `${hh}:${m.toString().padStart(2, "0")} ${mer}`;
  };
  if (p.session_end) return `${fmt(p.session_start)}–${fmt(p.session_end)}`;
  return fmt(p.session_start);
}

export function firstAuthors(p: Paper, max: number = 3): string {
  const a = p.authors || [];
  if (a.length <= max) return a.join(", ");
  return a.slice(0, max).join(", ") + `, +${a.length - max}`;
}
