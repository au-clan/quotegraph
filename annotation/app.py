"""
Quotation Verification Annotation Interface

A Streamlit-based tool for verifying quotation extraction, attribution,
and Wikidata entity linking in news articles.
"""

import streamlit as st
import json
import re
import sys
import time
import uuid
from pathlib import Path
from datetime import datetime

# Configuration
DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "input_data.json"
ANNOTATIONS_DIR = DATA_DIR / "annotations"

# Color scheme
COLOR_QUOTATION = "#FFEB3B"  # Yellow for quotation
COLOR_SPEAKER = "#81D4FA"    # Light blue for speaker mentions
COLOR_MENTION = "#C8E6C9"    # Light green for person mentions

# Test mode: enable with --test flag
# Usage: streamlit run app.py -- --test
TEST_MODE = "--test" in sys.argv


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    return f"{minutes}m {seconds % 60:.0f}s"


def load_input_data():
    """Load the input JSON file containing quotations to annotate."""
    if not INPUT_FILE.exists():
        st.error(f"Input file not found: {INPUT_FILE}")
        st.info("Please create an input_data.json file in the data/ directory.")
        return []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def load_annotations(annotator_id: str) -> dict:
    """Load existing annotations for a specific annotator."""
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    annotation_file = ANNOTATIONS_DIR / f"annotations_{annotator_id}.json"

    if annotation_file.exists():
        with open(annotation_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_annotations(annotator_id: str, annotations: dict):
    """Save annotations for a specific annotator."""
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    annotation_file = ANNOTATIONS_DIR / f"annotations_{annotator_id}.json"

    with open(annotation_file, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

def get_wikidata_url(qid: str) -> str:
    """Generate Wikidata URL from QID."""
    if qid and qid.startswith("Q"):
        return f"https://www.wikidata.org/wiki/{qid}"
    return None

def highlight_context(context: str, quotation: str, speaker_name: str, mentions: list) -> str:
    """
    Highlight the context with:
    - Color 1 (yellow): quotation
    - Color 2 (blue): speaker mentions
    - Color 3 (green): person mentions (underlined in quotation)
    """
    if not context:
        return context

    result = context

    # Prepare quotation with underlined mentions
    quotation_with_underlined_mentions = quotation
    for mention in mentions:
        mention_name = mention.get("name", "")
        if mention_name and mention_name in quotation_with_underlined_mentions:
            quotation_with_underlined_mentions = quotation_with_underlined_mentions.replace(
                mention_name,
                f'<u style="text-decoration: underline; text-decoration-color: #333; text-underline-offset: 3px;">{mention_name}</u>'
            )

    # Use regex to find the quotation in context, allowing for flexible punctuation
    # Strip trailing punctuation from quotation for matching
    quotation_base = quotation.rstrip('.,!?;:')

    # Pattern to match quoted text with flexible ending punctuation
    # Matches: "quotation" or "quotation," or "quotation." etc.
    pattern = re.compile(
        r'"(' + re.escape(quotation_base) + r')[.,!?;:]*"',
        re.IGNORECASE
    )

    match = pattern.search(result)
    if match:
        # Replace the matched quotation with highlighted version
        original = match.group(0)
        highlighted = f'<mark style="background-color: {COLOR_QUOTATION}; padding: 2px 4px; border-radius: 3px;">"{quotation_with_underlined_mentions}"</mark>'
        result = result.replace(original, highlighted, 1)
    else:
        # Fallback: try exact match without quotes
        if quotation and quotation in result:
            result = result.replace(
                quotation,
                f'<mark style="background-color: {COLOR_QUOTATION}; padding: 2px 4px; border-radius: 3px;">{quotation_with_underlined_mentions}</mark>',
                1
            )

    # Highlight speaker name mentions (outside of the already highlighted quotation)
    if speaker_name:
        # Use regex to find speaker name not inside already processed tags
        pattern = re.compile(rf'(?<!</mark>)(?<!</span>)(?<!</u>)(?<!">)\b({re.escape(speaker_name)})\b(?!<)')
        result = pattern.sub(
            rf'<span style="background-color: {COLOR_SPEAKER}; padding: 2px 4px; border-radius: 3px;">\1</span>',
            result
        )

    # Highlight person mentions in the context (outside quotation)
    for mention in mentions:
        mention_name = mention.get("name", "")
        if mention_name:
            # Find mentions not already inside tags
            pattern = re.compile(rf'(?<!</mark>)(?<!</span>)(?<!</u>)(?<!">)\b({re.escape(mention_name)})\b(?!<)')
            result = pattern.sub(
                rf'<span style="background-color: {COLOR_MENTION}; padding: 2px 4px; border-radius: 3px;">\1</span>',
                result
            )

    return result

def format_quotation_with_mentions(quotation: str, mentions: list) -> str:
    """Underline mentions inside the quotation."""
    if not quotation or not mentions:
        return quotation

    result = quotation
    for mention in mentions:
        mention_name = mention.get("name", "")
        if mention_name and mention_name in result:
            result = result.replace(
                mention_name,
                f'<u style="text-decoration-color: #333;">{mention_name}</u>'
            )

    return result

def display_entity_with_wikidata(name: str, wikidata_id: str, description: str = None):
    """Display an entity name with a link to its Wikidata page and description."""
    url = get_wikidata_url(wikidata_id)
    if url:
        st.markdown(f"**{name}** ([{wikidata_id}]({url}))")
        if description:
            st.caption(f"_{description}_")
    else:
        st.markdown(f"**{name}** (No Wikidata ID)")

def render_annotation_form(item: dict, item_id: str, existing_annotation: dict):
    """Render the annotation form for a single item."""

    quotation = item.get("quotation", "")
    context = item.get("context", "")
    speaker = item.get("speaker", {})
    speaker_name = speaker.get("name", "Unknown")
    mentions = item.get("mentions", [])

    # Two-column layout
    col_data, col_questions = st.columns([1, 1], gap="large")

    # LEFT COLUMN: Data
    with col_data:
        # Display the quotation with underlined mentions
        st.markdown("**Quotation**")
        formatted_quotation = format_quotation_with_mentions(quotation, mentions)
        st.markdown(
            f'<div style="font-size: 1.1em; color: #333; background-color: #fafafa; padding: 12px; border-left: 4px solid {COLOR_QUOTATION}; margin-bottom: 15px;">"{formatted_quotation}"</div>',
            unsafe_allow_html=True
        )

        # Display the context with highlighted quotation and speaker
        st.markdown("**Article Context**")
        highlighted_context = highlight_context(context, quotation, speaker_name, mentions)
        st.markdown(
            f'<div style="font-size: 1em; color: #333; background-color: #fafafa; padding: 15px; border-radius: 5px; border: 1px solid #ddd; line-height: 1.6; max-height: 300px; overflow-y: auto;">{highlighted_context}</div>',
            unsafe_allow_html=True
        )

        # Legend
        st.markdown(
            f'''<div style="font-size: 0.85em; color: #666; margin: 8px 0;">
            <span style="background-color: {COLOR_QUOTATION}; padding: 2px 5px; border-radius: 3px; margin-right: 10px;">Quotation</span>
            <span style="background-color: {COLOR_SPEAKER}; padding: 2px 5px; border-radius: 3px; margin-right: 10px;">Speaker</span>
            <span style="background-color: {COLOR_MENTION}; padding: 2px 5px; border-radius: 3px; margin-right: 10px;">Mention</span>
            <span style="text-decoration: underline;">In quote</span>
            </div>''',
            unsafe_allow_html=True
        )

        # Display the speaker
        st.markdown("**Attributed Speaker**")
        speaker_url = get_wikidata_url(speaker.get("wikidata_id", ""))
        if speaker_url:
            st.markdown(f"{speaker_name} ([{speaker.get('wikidata_id')}]({speaker_url}))")
        else:
            st.markdown(f"{speaker_name}")

        # Display mentioned persons
        if mentions:
            st.markdown("**Mentioned Persons**")
            for mention in mentions:
                mention_url = get_wikidata_url(mention.get("wikidata_id", ""))
                if mention_url:
                    st.markdown(f"- {mention.get('name')} ([{mention.get('wikidata_id')}]({mention_url}))")
                else:
                    st.markdown(f"- {mention.get('name')}")

    # RIGHT COLUMN: Questions
    with col_questions:
        st.markdown("**Annotation Questions**")

        # Q1: Speaker attribution
        st.markdown("**Q1:** Is the provided quotation attributed to the correct speaker?")
        q1_options = ["Select...", "Yes", "No", "Cannot determine"]
        q1_default = q1_options.index(existing_annotation.get("q1", "Select...")) if existing_annotation.get("q1") in q1_options else 0
        q1_answer = st.radio(
            "Q1",
            options=q1_options,
            index=q1_default,
            key=f"q1_{item_id}",
            horizontal=True,
            label_visibility="collapsed"
        )

        # Q2: Speaker Wikidata linking (only if Q1 is Yes)
        q2_answer = None
        if q1_answer == "Yes":
            st.markdown("**Q2:** Is the speaker correctly linked to its corresponding Wikidata item based on the information provided in the article context?")
            speaker_wikidata_id = speaker.get("wikidata_id", "")
            speaker_description = speaker.get("description", "No description available")
            speaker_url = get_wikidata_url(speaker_wikidata_id)

            if speaker_url:
                st.markdown(
                    f'<div style="font-size: 0.95em; color: #333; background-color: #f0f7ff; padding: 8px; border-radius: 4px; margin: 5px 0;">'
                    f'<strong>{speaker_name}</strong> → <a href="{speaker_url}" target="_blank">{speaker_wikidata_id}</a><br/>'
                    f'<em style="color: #666;">{speaker_description}</em></div>',
                    unsafe_allow_html=True
                )

            q2_options = ["Select...", "Yes", "No", "Cannot determine"]
            q2_default = q2_options.index(existing_annotation.get("q2", "Select...")) if existing_annotation.get("q2") in q2_options else 0
            q2_answer = st.radio(
                "Q2",
                options=q2_options,
                index=q2_default,
                key=f"q2_{item_id}",
                horizontal=True,
                label_visibility="collapsed"
            )

        # Q3 and Q4: Only if there are mentions
        q3_answers = {}
        q4_answers = {}

        if mentions:
            st.markdown("**Q3:** Do the listed mentions refer to people and not other objects, organizations or locations?")
            for i, mention in enumerate(mentions):
                mention_name = mention.get("name", f"Mention {i+1}")
                mention_id = mention.get("id", str(i))

                q3_options = ["Select...", "Yes", "No", "Cannot determine"]
                q3_key = f"q3_{item_id}_{mention_id}"
                existing_q3 = existing_annotation.get("q3", {}).get(mention_id, "Select...")
                q3_default = q3_options.index(existing_q3) if existing_q3 in q3_options else 0

                q3_answers[mention_id] = st.radio(
                    f"'{mention_name}'",
                    options=q3_options,
                    index=q3_default,
                    key=q3_key,
                    horizontal=True
                )

            st.markdown("**Q4:** Is each mentioned person correctly linked to their corresponding Wikidata items?")

            for i, mention in enumerate(mentions):
                mention_name = mention.get("name", f"Mention {i+1}")
                mention_id = mention.get("id", str(i))
                mention_wikidata_id = mention.get("wikidata_id", "")
                mention_description = mention.get("description", "No description available")

                # Only show Q4 if Q3 indicates it's a person
                if q3_answers.get(mention_id) == "Yes":
                    mention_url = get_wikidata_url(mention_wikidata_id)

                    if mention_url:
                        st.markdown(
                            f'<div style="font-size: 0.95em; color: #333; background-color: #f0f7ff; padding: 8px; border-radius: 4px; margin: 5px 0;">'
                            f'<strong>{mention_name}</strong> → <a href="{mention_url}" target="_blank">{mention_wikidata_id}</a><br/>'
                            f'<em style="color: #666;">{mention_description}</em></div>',
                            unsafe_allow_html=True
                        )

                    q4_options = ["Select...", "Yes", "No", "Cannot determine"]
                    q4_key = f"q4_{item_id}_{mention_id}"
                    existing_q4 = existing_annotation.get("q4", {}).get(mention_id, "Select...")
                    q4_default = q4_options.index(existing_q4) if existing_q4 in q4_options else 0

                    q4_answers[mention_id] = st.radio(
                        f"'{mention_name}'",
                        options=q4_options,
                        index=q4_default,
                        key=q4_key,
                        horizontal=True
                    )

        # Optional notes
        st.markdown("**Notes** (optional)")
        notes = st.text_area(
            "Notes",
            value=existing_annotation.get("notes", ""),
            key=f"notes_{item_id}",
            height=80,
            label_visibility="collapsed",
            placeholder="Add any comments..."
        )

    return {
        "q1": q1_answer if q1_answer != "Select..." else None,
        "q2": q2_answer if q2_answer and q2_answer != "Select..." else None,
        "q3": {k: v for k, v in q3_answers.items() if v != "Select..."},
        "q4": {k: v for k, v in q4_answers.items() if v != "Select..."},
        "notes": notes,
        "timestamp": datetime.now().isoformat()
    }

def is_annotation_complete(annotation: dict, item: dict) -> bool:
    """Check if an annotation is complete (all required questions answered)."""
    if not annotation.get("q1"):
        return False

    if annotation["q1"] == "Yes" and not annotation.get("q2"):
        return False

    mentions = item.get("mentions", [])
    if mentions:
        q3 = annotation.get("q3", {})
        q4 = annotation.get("q4", {})

        for mention in mentions:
            mention_id = mention.get("id", str(mentions.index(mention)))
            if mention_id not in q3:
                return False
            if q3.get(mention_id) == "Yes" and mention_id not in q4:
                return False

    return True

def main():
    st.set_page_config(
        page_title="Quotation Verification Annotator",
        page_icon="📝",
        layout="wide"
    )

    st.title("Quotation Verification Annotation Interface")
    st.markdown("Verify quotation extraction, speaker attribution, and Wikidata entity linking.")

    # Sidebar for annotator info and navigation
    with st.sidebar:
        st.header("Annotator Settings")

        # Generate an ID on first visit so the annotator can start immediately
        if "annotator_id" not in st.session_state:
            st.session_state.annotator_id = uuid.uuid4().hex[:8]

        annotator_id = st.text_input(
            "Annotator ID",
            value=st.session_state.annotator_id,
            placeholder="Enter your annotator ID"
        )

        if annotator_id:
            st.session_state.annotator_id = annotator_id
            st.success(f"Logged in as: {annotator_id}")
            if TEST_MODE:
                st.info("⏱ Test mode active")
        else:
            st.warning("Please enter your annotator ID to begin.")
            st.stop()

        st.divider()

        # Load data
        data = load_input_data()
        if not data:
            st.stop()

        annotations = load_annotations(annotator_id)

        # Progress tracking
        st.header("Progress")
        completed = sum(1 for i, item in enumerate(data)
                       if is_annotation_complete(annotations.get(str(i), {}), item))
        progress = completed / len(data) if data else 0
        st.progress(progress)
        st.write(f"Completed: {completed} / {len(data)}")

        st.divider()

        # Navigation
        st.header("Navigation")

        # Item selector
        item_options = [f"Item {i+1}" + (" ✓" if is_annotation_complete(annotations.get(str(i), {}), item) else "")
                       for i, item in enumerate(data)]

        current_index = st.session_state.get("current_index", 0)
        selected_item = st.selectbox(
            "Select item",
            options=range(len(data)),
            format_func=lambda x: item_options[x],
            index=current_index
        )
        st.session_state.current_index = selected_item

        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Previous", disabled=selected_item == 0, use_container_width=True):
                st.session_state.current_index = selected_item - 1
                st.rerun()
        with col2:
            if st.button("Next →", disabled=selected_item == len(data) - 1, use_container_width=True):
                st.session_state.current_index = selected_item + 1
                st.rerun()

        st.divider()

        # Export button
        st.header("Export")
        if st.button("Export Annotations", use_container_width=True):
            export_data = {
                "annotator_id": annotator_id,
                "export_timestamp": datetime.now().isoformat(),
                "total_items": len(data),
                "completed_items": completed,
                "annotations": annotations
            }
            st.download_button(
                label="Download JSON",
                data=json.dumps(export_data, indent=2, ensure_ascii=False),
                file_name=f"annotations_{annotator_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

        if TEST_MODE:
            st.divider()
            st.header("Timing")
            durations = st.session_state.get("batch_durations", {})
            if durations:
                for idx, dur in sorted(durations.items(), key=lambda x: int(x[0])):
                    st.write(f"Item {int(idx) + 1}: {format_duration(dur)}")
                avg = sum(durations.values()) / len(durations)
                st.write(f"**Average: {format_duration(avg)}**")
            else:
                st.write("No items timed yet.")

    # Main content area
    current_item = data[selected_item]
    item_id = str(selected_item)
    existing_annotation = annotations.get(item_id, {})

    # Test mode: record when the annotator first views each item
    if TEST_MODE:
        if "batch_start_times" not in st.session_state:
            st.session_state.batch_start_times = {}
        if "batch_durations" not in st.session_state:
            st.session_state.batch_durations = {}
        if item_id not in st.session_state.batch_start_times:
            st.session_state.batch_start_times[item_id] = time.time()

    # Display item header
    st.markdown(f"### Item {selected_item + 1} of {len(data)}")
    if current_item.get("source"):
        st.caption(f"Source: {current_item.get('source')}")

    # Render the annotation form
    new_annotation = render_annotation_form(current_item, item_id, existing_annotation)

    # Save button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Save Annotation", type="primary", use_container_width=True):
            if TEST_MODE:
                start = st.session_state.get("batch_start_times", {}).get(item_id)
                if start is not None:
                    st.session_state.batch_durations[item_id] = time.time() - start
            annotations[item_id] = new_annotation
            save_annotations(annotator_id, annotations)
            st.success("Annotation saved!")

            # Check if complete
            if is_annotation_complete(new_annotation, current_item):
                st.balloons()

            st.rerun()

    # Show completion status
    if is_annotation_complete(existing_annotation, current_item):
        st.success("This item has been fully annotated.")
    else:
        st.info("Please answer all required questions to complete this annotation.")

if __name__ == "__main__":
    main()
