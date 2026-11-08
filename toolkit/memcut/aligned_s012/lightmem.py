"""LightMem's native tail, observed rather than changed.

On the last turn MemCut calls add_memory(force_segment=True) as LoCoMo's add_locomo.py does. Upstream then
replaces the segments that call's add_messages has just produced with the forced cut of the buffer
(lightmem.py:312-315), so those turns are never extracted. The aligned entry keeps that behaviour (the cloud
runner's optional flush_last repair is off there too) and records the turns it drops in
progress['diagnostics']['final_segments_dropped_turns'], as the cloud runner counts final_dropped.
"""
from memcut.source_anchored_systems.formation_runtime.lightmem import FastLightMemBank

WATCHED = ('add_messages', 'cut_with_segmenter')


class AlignedLightMemBank(FastLightMemBank):
    def ingest(self, turn_id, completions, *, final=False):
        if not final:
            return super().ingest(turn_id, completions, final=final)
        buffers = self.memory.senmem_buffer_manager
        own = {name: buffers.__dict__[name] for name in WATCHED if name in buffers.__dict__}
        add_messages, cut_with_segmenter = buffers.add_messages, buffers.cut_with_segmenter
        produced, dropped = [], []

        def add_messages_seen(messages, segmenter, text_embedder):
            segments = add_messages(messages, segmenter, text_embedder)
            produced[:] = segments
            return segments

        def cut_final(segmenter, text_embedder, force_segment=False):
            segments = cut_with_segmenter(segmenter, text_embedder, force_segment)
            if force_segment and produced:
                dropped.extend(m.get('_memcut_source_id') for segment in produced for m in segment
                               if m.get('role') == 'user')
                produced.clear()
            return segments

        buffers.add_messages, buffers.cut_with_segmenter = add_messages_seen, cut_final
        try:
            return super().ingest(turn_id, completions, final=True)
        finally:
            for name in WATCHED:
                if name in own:
                    setattr(buffers, name, own[name])
                else:
                    delattr(buffers, name)
            diagnostics = self.progress.setdefault('diagnostics', {})
            diagnostics['final_segments_dropped_turns'] = list(dict.fromkeys(t for t in dropped if t))
