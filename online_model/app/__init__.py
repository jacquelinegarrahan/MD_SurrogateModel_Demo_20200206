from argparse import ArgumentParser


def get_protocol():
    # Parse arguments passed through bokeh serve
    # requires protocol to be set
    parser = ArgumentParser()
    parser.add_argument(
        "-p",
        "--protocol",
        metavar="PROTOCOL",
        nargs=1,
        type=str,
        choices=["pva", "ca"],
        help="Protocol to use (ca, pva)",
        required=True,
    )
    args = parser.parse_args()

    # get protocol
    return args.protocol[0]
