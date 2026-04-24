import { papers, enriched, topicBySlug } from "./data";

export interface SearchDoc {
  id: string;
  title: string;
  authors: string;           // joined for MiniSearch tokenization
  authors_display: string;   // first few authors, comma-joined, for UI
  tldr: string;
  summary: string;
  keywords: string;
  topic: string;
  room: string;
  date: string;
  arxiv_id: string;
}

function abbreviateAuthors(authors: string[], max = 3): string {
  if (authors.length <= max) return authors.join(", ");
  return authors.slice(0, max).join(", ") + `, +${authors.length - max}`;
}

export const documents: SearchDoc[] = papers.map((p) => {
  const e = enriched[p.id];
  return {
    id: p.id,
    title: p.title,
    authors: p.authors.join(" "),
    authors_display: abbreviateAuthors(p.authors),
    tldr: p.tldr ?? "",
    summary: e?.one_sentence_summary ?? "",
    keywords: p.keywords.join(" "),
    topic: topicBySlug.get(p.topic_slug)?.label ?? p.topic_slug,
    room: p.room ?? "",
    date: p.session_date ?? "",
    arxiv_id: p.arxiv_id ?? "",
  };
});
