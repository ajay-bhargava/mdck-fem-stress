"""Pre-render pos01 stress frames using the series-wide fixed color scale.

Run: uv run python -m scripts.build_stress_display
"""
from pathlib import Path
from src.web.data_service import DataService


def main():
    root=Path(__file__).resolve().parents[1]/'data'
    service=DataService(root)
    summary=service.stress_series_summary('pos01')
    import json
    output=root/'pos01/fem/stress/time_series/display_thickness_v1'
    output.mkdir(exist_ok=True)
    for thickness in summary['thickness_options_um']:
        folder=output/f'{thickness:g}um';folder.mkdir(exist_ok=True)
        for frame in range(summary['frame_count']):
            path=folder/f'frame{frame:03d}.png'
            if not path.exists():
                temporary=path.with_suffix('.tmp')
                temporary.write_bytes(service.stress_series_png('pos01',frame,thickness))
                temporary.replace(path)
        print(f'{thickness:g} um: {summary["frame_count"]} frames ready',flush=True)
    (output/'provenance.json').write_text(json.dumps({
        'source':'../summary.json','reference_thickness_m':summary['assumptions']['thickness_m'],
        'options_um':summary['thickness_options_um'],
        'method':summary['thickness_scaling'],
        'fixed_color_max_pa':summary['thickness_comparison_color_max_pa'],
        'full_fields':'Stored reference tensors multiplied by h_reference/h; no duplicated solve arrays.',
    },indent=2)+'\n')

if __name__=='__main__':main()
