import { papers, enriched, topicBySlug } from "./data";

export interface SearchDoc {
  id: string;
  title: string;
  authors: string;           // joined for MiniSearch tokenization
  tldr: string;
  summary: string;
  keywords: string;
  topic: string;
  room: string;
  date: string;
}

export const documents: SearchDoc[] = papers.map((p) => {
  const e = enriched[p.id];
  return {
    id: p.id,
    title: p.title,
    authors: p.authors.join(" "),
    tldr: p.tldr ?? "",
    summary: e?.one_sentence_summary ?? "",
    keywords: p.keywords.join(" "),
    topic: topicBySlug.get(p.topic_slug)?.label ?? p.topic_slug,
    room: p.room ?? "",
    date: p.session_date ?? "",
  };
});
