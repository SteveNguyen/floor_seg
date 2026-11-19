"""Record the home owner's (or tenant's) written consent for a staged session.

    python -m floor_seg.dataset.consent STAGING/home03 --given y --date 2026-11-20 --role owner \
        --form-ref "signed paper form, binder B, page 3" [--register ~/.local/share/floor_seg/consent_register.jsonl]

The consent form is docs/dataset/CONSENT.md. In the session (and so possibly in what is exported) only
{given, date, signer_role} is stored: no names. Where the signed form is kept (--form-ref) goes into a separate
local register, outside the session and the dataset (default: $XDG_DATA_HOME/floor_seg/consent_register.jsonl);
keep it private, it is what links a session id to a home. --given n records a refusal: export then refuses the
session (delete the session folder). A withdrawal later: re-run with --given n, delete the session and rebuild
the dataset without it.
"""
import argparse
import datetime
import json
import os

from floor_seg.dataset import load_session, save_session

ROLES = ("owner", "tenant")


def default_register():
    base = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
    return os.path.join(base, "floor_seg", "consent_register.jsonl")


def record(session_dir, given, date, role, form_ref, register):
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    datetime.date.fromisoformat(date)  # YYYY-MM-DD, raises otherwise
    # the register must not end up inside the session (it would be next to what gets exported)
    sess, reg = os.path.realpath(session_dir), os.path.realpath(register)
    if os.path.commonpath([sess, reg]) == sess:
        raise ValueError(f"the register {register} must be outside the session folder")
    session = load_session(session_dir)
    session["consent"] = {"given": bool(given), "date": date, "signer_role": role}
    save_session(session_dir, session)
    os.makedirs(os.path.dirname(reg), exist_ok=True)
    with open(reg, "a") as f:
        f.write(json.dumps({"session_id": session["session_id"], "given": bool(given), "date": date,
                            "signer_role": role, "form_ref": form_ref}) + "\n")
    return session["consent"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", help="staged session folder")
    ap.add_argument("--given", required=True, choices=["y", "n"], help="written consent given")
    ap.add_argument("--date", required=True, help="date of the signature, YYYY-MM-DD")
    ap.add_argument("--role", required=True, choices=ROLES, help="who signed")
    ap.add_argument("--form-ref", required=True, help="where the signed form is kept (register only)")
    ap.add_argument("--register", default=default_register(), help="local consent register (outside the dataset)")
    args = ap.parse_args()
    c = record(args.session, args.given == "y", args.date, args.role, args.form_ref, args.register)
    print(f"{args.session}: consent {'given' if c['given'] else 'REFUSED'} ({c['signer_role']}, {c['date']}); "
          f"form reference in {args.register}")


if __name__ == "__main__":
    main()
