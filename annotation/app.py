"""
Quotation Verification Annotation Interface
"""

import streamlit as st
import json
import ast
import html
import re
import sys
import time
import uuid
from pathlib import Path
from datetime import datetime

# Configuration
TEST_BATCH_FILE = Path("/home/mculjak/jadranka/test_batch.json")
DATA_DIR = Path("data")
ANNOTATIONS_DIR = DATA_DIR / "annotations"

# Color scheme
COLOR_QUOTATION = "#FFEB3B"
COLOR_SPEAKER   = "#81D4FA"
COLOR_TARGET    = "#CE93D8"
COLOR_MENTION   = "#C8E6C9"

TEST_MODE = "--test" in sys.argv


def format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m {seconds % 60:.0f}s"


def get_wikidata_url(qid):
    if qid and str(qid).startswith("Q"):
        return f"https://www.wikidata.org/wiki/{qid}"
    return None


def load_and_transform():
    """Load test_batch.json and return list of annotation items (one per article, first quotation only)."""
    if not TEST_BATCH_FILE.exists():
        st.error(f"Test batch file not found: {TEST_BATCH_FILE}")
        return []
    with open(TEST_BATCH_FILE, encoding="utf-8") as f:
        articles = json.load(f)

    items = []
    for article in articles:
        quotations = article.get("quotations") or []
        if not quotations:
            continue
        quotation = quotations[0]

        # Speaker QID: prefer edge.speaker.qid, fallback to globalProbas
        speaker_edge = (quotation.get("edge") or {}).get("speaker") or {}
        speaker_qid = speaker_edge.get("qid")
        if not speaker_qid:
            top_speaker = quotation.get("globalTopSpeaker")
            for prob in quotation.get("globalProbas") or []:
                if prob.get("speaker") == top_speaker and prob.get("qids"):
                    speaker_qid = prob["qids"][0]
                    break

        target_edge = (quotation.get("edge") or {}).get("target") or {}
        target_qid = target_edge.get("qid")

        # Classify names: speaker / target / other
        names = article.get("names") or []
        speaker_names, target_names, mention_names = [], [], []
        for name in names:
            try:
                ids = ast.literal_eval(name.get("ids", "[]"))
            except Exception:
                ids = []
            if speaker_qid and speaker_qid in ids:
                speaker_names.append(name)
            elif target_qid and target_qid in ids:
                target_names.append(name)
            else:
                mention_names.append(name)

        items.append({
            "article_id": article.get("articleID", ""),
            "title": article.get("title", ""),
            "url": article.get("url", ""),
            "detokenized_content": article.get("detokenized_content", ""),
            "quotation": quotation.get("quotation", ""),
            "char_start": quotation.get("charStart", -1),
            "char_end": quotation.get("charEnd", -1),
            "context_char_start": quotation.get("contextCharStart", -1),
            "context_char_end": quotation.get("contextCharEnd", -1),
            "speaker": {
                "qid": speaker_qid,
                "label": speaker_edge.get("wikidata_label") or quotation.get("globalTopSpeaker") or "Unknown",
                "description": speaker_edge.get("wikidata_description", ""),
            },
            "target": {
                "qid": target_qid,
                "label": target_edge.get("wikidata_label", ""),
                "description": target_edge.get("wikidata_description", ""),
            },
            "speaker_names": speaker_names,
            "target_names": target_names,
            "mention_names": mention_names,
        })
    return items


def build_highlighted_context(item):
    """Build HTML for the context window using character offsets."""
    text = item["detokenized_content"]
    ctx_s = item["context_char_start"]
    ctx_e = item["context_char_end"]

    if ctx_s == -1 or ctx_e == -1 or not text:
        return html.escape(item.get("quotation", ""))

    context = text[ctx_s:ctx_e]
    n = len(context)
    labels = [""] * n

    def mark(char_spans, label):
        for s, e in char_spans:
            for i in range(max(0, s - ctx_s), min(n, e - ctx_s)):
                if not labels[i]:
                    labels[i] = label

    mark([(item["char_start"], item["char_end"])], "quote")
    for nm in item["speaker_names"]:
        mark(nm.get("char_offsets", []), "speaker")
    for nm in item["target_names"]:
        mark(nm.get("char_offsets", []), "target")
    for nm in item["mention_names"]:
        mark(nm.get("char_offsets", []), "mention")

    COLOR_MAP = {"quote": COLOR_QUOTATION, "speaker": COLOR_SPEAKER,
                 "target": COLOR_TARGET, "mention": COLOR_MENTION}
    parts = []
    i = 0
    while i < n:
        label = labels[i]
        j = i + 1
        while j < n and labels[j] == label:
            j += 1
        chunk = html.escape(context[i:j])
        if label:
            c = COLOR_MAP[label]
            parts.append(f'<span style="background-color:{c};padding:1px 2px;border-radius:2px;">{chunk}</span>')
        else:
            parts.append(chunk)
        i = j
    return "".join(parts)


def load_annotations(annotator_id):
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = ANNOTATIONS_DIR / f"annotations_{annotator_id}.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_annotations(annotator_id, annotations):
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = ANNOTATIONS_DIR / f"annotations_{annotator_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)


def render_annotation_form(item, item_id, existing):
    col_data, col_q = st.columns([1, 1], gap="large")

    with col_data:
        st.markdown("**Quotation**")
        st.markdown(
            f'<div style="font-size:1.1em;background:#fafafa;padding:12px;'
            f'border-left:4px solid {COLOR_QUOTATION};margin-bottom:12px;">'
            f'"{html.escape(item["quotation"])}"</div>',
            unsafe_allow_html=True
        )

        st.markdown("**Article Context**")
        st.markdown(
            f'<div style="font-size:1em;background:#fafafa;padding:15px;border-radius:5px;'
            f'border:1px solid #ddd;line-height:1.7;max-height:340px;overflow-y:auto;">'
            f'{build_highlighted_context(item)}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div style="font-size:0.85em;color:#666;margin:6px 0;">'
            f'<span style="background:{COLOR_QUOTATION};padding:2px 5px;border-radius:3px;margin-right:8px;">Quotation</span>'
            f'<span style="background:{COLOR_SPEAKER};padding:2px 5px;border-radius:3px;margin-right:8px;">Speaker</span>'
            f'<span style="background:{COLOR_TARGET};padding:2px 5px;border-radius:3px;margin-right:8px;">Target</span>'
            f'<span style="background:{COLOR_MENTION};padding:2px 5px;border-radius:3px;">Other mentions</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Speaker info
        speaker = item["speaker"]
        speaker_url = get_wikidata_url(speaker["qid"])
        st.markdown("**Speaker**")
        if speaker_url:
            st.markdown(f"{speaker['label']} ([{speaker['qid']}]({speaker_url}))")
        else:
            st.markdown(speaker["label"])
        if speaker["description"]:
            st.caption(f"_{speaker['description']}_")

        # Target info
        target = item["target"]
        if target["label"]:
            target_url = get_wikidata_url(target["qid"])
            st.markdown("**Target**")
            if target_url:
                st.markdown(f"{target['label']} ([{target['qid']}]({target_url}))")
            else:
                st.markdown(target["label"])
            if target["description"]:
                st.caption(f"_{target['description']}_")

        # All identified persons
        all_names = item["speaker_names"] + item["target_names"] + item["mention_names"]
        if all_names:
            st.markdown("**Identified Persons**")
            for nm in all_names:
                try:
                    ids = ast.literal_eval(nm.get("ids", "[]"))
                except Exception:
                    ids = []
                qid = ids[0] if ids else None
                url = get_wikidata_url(qid)
                label = f"{nm['name']} ([{qid}]({url}))" if url else nm["name"]
                st.markdown(f"- {label}")

    with col_q:
        st.markdown("**Annotation Questions**")

        # Q1
        st.markdown("**Q1:** Is the provided quotation attributed to the correct speaker?")
        q1_opts = ["Select...", "Yes", "No", "Cannot determine"]
        q1_default = q1_opts.index(existing.get("q1", "Select...")) if existing.get("q1") in q1_opts else 0
        q1 = st.radio("Q1", q1_opts, index=q1_default, key=f"q1_{item_id}", horizontal=True, label_visibility="collapsed")

        # Q2 (only if Q1 = Yes)
        q2 = None
        if q1 == "Yes":
            st.markdown("**Q2:** Is the speaker correctly linked to its corresponding Wikidata item?")
            if speaker_url:
                st.markdown(
                    f'<div style="font-size:0.95em;background:#f0f7ff;padding:8px;border-radius:4px;margin:5px 0;">'
                    f'<strong>{html.escape(speaker["label"])}</strong> → '
                    f'<a href="{speaker_url}" target="_blank">{speaker["qid"]}</a><br/>'
                    f'<em style="color:#666;">{html.escape(speaker["description"])}</em></div>',
                    unsafe_allow_html=True
                )
            q2_opts = ["Select...", "Yes", "No", "Cannot determine"]
            q2_default = q2_opts.index(existing.get("q2", "Select...")) if existing.get("q2") in q2_opts else 0
            q2 = st.radio("Q2", q2_opts, index=q2_default, key=f"q2_{item_id}", horizontal=True, label_visibility="collapsed")

        # Q3 & Q4 — all non-speaker names
        q3, q4 = {}, {}
        mentions = item["target_names"] + item["mention_names"]
        if mentions:
            st.markdown("**Q3:** Do the listed mentions refer to people and not other objects, organizations or locations?")
            for nm in mentions:
                mid = nm["name"]
                q3_opts = ["Select...", "Yes", "No", "Cannot determine"]
                existing_q3 = existing.get("q3", {}).get(mid, "Select...")
                q3_default = q3_opts.index(existing_q3) if existing_q3 in q3_opts else 0
                q3[mid] = st.radio(f"'{mid}'", q3_opts, index=q3_default, key=f"q3_{item_id}_{mid}", horizontal=True)

            st.markdown("**Q4:** Is each mentioned person correctly linked to their corresponding Wikidata items?")
            for nm in mentions:
                mid = nm["name"]
                if q3.get(mid) == "Yes":
                    try:
                        ids = ast.literal_eval(nm.get("ids", "[]"))
                    except Exception:
                        ids = []
                    qid = ids[0] if ids else None
                    url = get_wikidata_url(qid)
                    if url:
                        st.markdown(
                            f'<div style="font-size:0.95em;background:#f0f7ff;padding:8px;border-radius:4px;margin:5px 0;">'
                            f'<strong>{html.escape(mid)}</strong> → '
                            f'<a href="{url}" target="_blank">{qid}</a></div>',
                            unsafe_allow_html=True
                        )
                    q4_opts = ["Select...", "Yes", "No", "Cannot determine"]
                    existing_q4 = existing.get("q4", {}).get(mid, "Select...")
                    q4_default = q4_opts.index(existing_q4) if existing_q4 in q4_opts else 0
                    q4[mid] = st.radio(f"'{mid}'", q4_opts, index=q4_default, key=f"q4_{item_id}_{mid}", horizontal=True)

        st.markdown("**Notes** (optional)")
        notes = st.text_area("Notes", value=existing.get("notes", ""), key=f"notes_{item_id}",
                             height=80, label_visibility="collapsed", placeholder="Add any comments...")

    return {
        "q1": q1 if q1 != "Select..." else None,
        "q2": q2 if q2 and q2 != "Select..." else None,
        "q3": {k: v for k, v in q3.items() if v != "Select..."},
        "q4": {k: v for k, v in q4.items() if v != "Select..."},
        "notes": notes,
        "timestamp": datetime.now().isoformat(),
    }


def is_annotation_complete(annotation, item):
    if not annotation.get("q1"):
        return False
    if annotation["q1"] == "Yes" and not annotation.get("q2"):
        return False
    mentions = item.get("target_names", []) + item.get("mention_names", [])
    q3 = annotation.get("q3", {})
    q4 = annotation.get("q4", {})
    for nm in mentions:
        mid = nm["name"]
        if mid not in q3:
            return False
        if q3.get(mid) == "Yes" and mid not in q4:
            return False
    return True


def main():
    st.set_page_config(page_title="Quotation Verification Annotator", page_icon="📝", layout="wide")
    st.title("Quotation Verification Annotation Interface")
    st.markdown("Verify quotation extraction, speaker attribution, and Wikidata entity linking.")

    with st.sidebar:
        st.header("Annotator Settings")
        if "annotator_id" not in st.session_state:
            st.session_state.annotator_id = uuid.uuid4().hex[:8]
        annotator_id = st.text_input("Annotator ID", value=st.session_state.annotator_id)
        if annotator_id:
            st.session_state.annotator_id = annotator_id
            st.success(f"Logged in as: {annotator_id}")
            if TEST_MODE:
                st.info("⏱ Test mode active")
        else:
            st.warning("Please enter your annotator ID to begin.")
            st.stop()

        st.divider()

        if "data" not in st.session_state:
            st.session_state.data = load_and_transform()
        data = st.session_state.data
        if not data:
            st.stop()

        annotations = load_annotations(annotator_id)

        st.header("Progress")
        completed = sum(1 for i, item in enumerate(data) if is_annotation_complete(annotations.get(str(i), {}), item))
        st.progress(completed / len(data) if data else 0)
        st.write(f"Completed: {completed} / {len(data)}")

        st.divider()
        st.header("Navigation")
        current_index = min(st.session_state.get("current_index", 0), len(data) - 1)
        item_options = [
            f"Item {i+1}: {item['title'][:35] or item['article_id']}" +
            (" ✓" if is_annotation_complete(annotations.get(str(i), {}), item) else "")
            for i, item in enumerate(data)
        ]
        selected = st.selectbox("Select item", options=range(len(data)),
                                format_func=lambda x: item_options[x], index=current_index) or 0
        st.session_state.current_index = selected

        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Prev", disabled=selected == 0, use_container_width=True):
                st.session_state.current_index = selected - 1
                st.rerun()
        with col2:
            if st.button("Next →", disabled=selected == len(data) - 1, use_container_width=True):
                st.session_state.current_index = selected + 1
                st.rerun()

        st.divider()
        st.header("Export")
        if st.button("Export Annotations", use_container_width=True):
            export_data = {
                "annotator_id": annotator_id,
                "export_timestamp": datetime.now().isoformat(),
                "total_items": len(data),
                "completed_items": completed,
                "annotations": annotations,
            }
            st.download_button(
                label="Download JSON",
                data=json.dumps(export_data, indent=2, ensure_ascii=False),
                file_name=f"annotations_{annotator_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )

        if TEST_MODE:
            st.divider()
            st.header("Timing")
            durations = st.session_state.get("batch_durations", {})
            if durations:
                for idx, dur in sorted(durations.items(), key=lambda x: int(x[0])):
                    st.write(f"Item {int(idx)+1}: {format_duration(dur)}")
                st.write(f"**Average: {format_duration(sum(durations.values()) / len(durations))}**")
            else:
                st.write("No items timed yet.")

    if selected >= len(data):
        st.error("No items to display.")
        st.stop()
    current_item = data[selected]
    item_id = str(selected)
    existing = annotations.get(item_id, {})

    if TEST_MODE:
        for key in ("batch_start_times", "batch_durations"):
            if key not in st.session_state:
                st.session_state[key] = {}
        if item_id not in st.session_state.batch_start_times:
            st.session_state.batch_start_times[item_id] = time.time()

    st.markdown(f"### Item {selected + 1} of {len(data)}: {current_item['title']}")
    if current_item.get("url"):
        st.caption(f"Source: {current_item['url']}")

    new_annotation = render_annotation_form(current_item, item_id, existing)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Save Annotation", type="primary", use_container_width=True):
            if TEST_MODE:
                start = st.session_state.get("batch_start_times", {}).get(item_id)
                if start:
                    st.session_state.batch_durations[item_id] = time.time() - start
            annotations[item_id] = new_annotation
            save_annotations(annotator_id, annotations)
            st.success("Annotation saved!")
            if is_annotation_complete(new_annotation, current_item):
                st.balloons()
            st.rerun()

    if is_annotation_complete(existing, current_item):
        st.success("This item has been fully annotated.")
    else:
        st.info("Please answer all required questions to complete this annotation.")


if __name__ == "__main__":
    main()
