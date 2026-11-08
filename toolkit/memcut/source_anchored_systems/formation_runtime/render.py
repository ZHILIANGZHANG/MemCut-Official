"""Pure native extraction-message rendering; no measurement CLI dependency."""
def render(arguments, index):
    parts = []
    topics = arguments.get('topic_id_mapping') or []
    ids = topics[index] if index < len(topics) else []
    for j, segment in enumerate(arguments['extract_list'][index]):
        lines = []
        for m in segment:
            if m['role'] != 'user':
                continue
            prefix = f"[{m['time_stamp']}, {m['weekday']}] " if m.get('time_stamp') and m.get('weekday') else ''
            lines.append(f"{prefix}{m['sequence_number']//2}.{m.get('speaker_name') or m['role']}: {m.get('content', '')}")
        parts.append(f"--- Topic {ids[j] if j < len(ids) else j+1} ---\n" + '\n'.join(lines))
    return [{'role': 'system', 'content': arguments['system_prompt']},
            {'role': 'user', 'content': '\n'.join(parts)}]
