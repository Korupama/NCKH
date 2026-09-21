"""Refine t0 from existing hybrid cache without rerunning detection or tracking."""
import argparse
import json
from ball_localization.contact import refine_contact_file

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('state-json','stage1-root','stage3-state','output-dir'):
        p.add_argument('--'+flag,required=True)
    p.add_argument('--stage4-handoff',help='Optional for FOOT; required for contact vertical plane')
    a=p.parse_args()
    result=refine_contact_file(a.state_json,a.stage1_root,a.stage3_state,a.stage4_handoff,a.output_dir)
    print(json.dumps(result['selected_frame_ball'],indent=2))

if __name__=='__main__': main()
