"""`python -m blueshadow <command> [args]`: one entry point for every step. Run it from the repository root."""
import runpy
import sys

COMMANDS = {  # command: (module, what it does)
    "fetch":    ("fetch_scene",     "download a Sentinel-1 (--sensor s1) or Sentinel-2 scene window from Planetary Computer"),
    "cut-ais":  ("cut_ais",         "cut a daily DMA AIS file to the minutes around a scene's acquisition time"),
    "sar":      ("pipeline_sar",    "run the pipeline on a SAR scene (Sentinel-1 or ICEYE): works at night and through cloud"),
    "s2":       ("pipeline_real",   "run the pipeline on a Sentinel-2 optical scene (daytime, cloud-free)"),
    "ui":       ("build_ui",        "build the single-file analyst page (ui.html) for a run folder given by OUT"),
    "link":     ("link_contacts",   "link contacts across several runs into tentative tracks"),
    "eval":     ("eval",            "score the pipeline on fake scenes with known ground truth"),
    "demo":     ("pipeline",        "run the original fully simulated demo scene"),
    "inspect":  ("inspect_data",    "unpack a downloaded Sentinel-2 .SAFE.zip and cut AIS to it"),
    "fake-s2":  ("make_fake_scene", "write the fake Sentinel-2 test scene to fixtures/s2"),
    "fake-sar": ("make_fake_sar",   "write a fake Sentinel-1 (--sensor s1) or ICEYE (--sensor iceye) scene to fixtures/"),
}


def usage():
    w = max(map(len, COMMANDS))
    lines = [f"  {c:<{w}}  {d}" for c, (_, d) in COMMANDS.items()]
    return "usage: python -m blueshadow <command> [--help | args]\n\ncommands:\n" + "\n".join(lines)


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(usage())
        return
    cmd = sys.argv[1]
    if cmd not in COMMANDS:
        sys.exit(f"unknown command {cmd!r}\n\n{usage()}")
    module = f"blueshadow.{COMMANDS[cmd][0]}"
    sys.argv = [f"python -m blueshadow {cmd}"] + sys.argv[2:]
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
