#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

# Copyright (C) 2022-2025 Gabriele Bozzola and Observables HQ
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, see <https://www.gnu.org/licenses/>.

import http.server
import logging
import socketserver
from string import Template

from kuibit import argparse_helper as kah
from kuibit.simdir import SimDir

html_template = """
<!DOCTYPE html>
<html>
  <head>
    <title>Timing information</title>
    <script src="https://unpkg.com/d3@7.0.4/dist/d3.min.js"></script>
  </head>
  <style>
      body {
        overflow: hidden;
      }
    </style>
  <body></body>
  <script>
    (function (d3$$1) {
      'use strict';

      function rectHeight(d) {
        return d.x1 - d.x0 - Math.min(1, (d.x1 - d.x0) / 2);
      }

      function labelVisible(d) {
        return d.y1 <= width && d.y0 >= 0 && d.x1 - d.x0 > 21;
      }

      const width = window.innerWidth;
      const height = window.innerHeight;

      function iciclePartition(data) {
        const root = d3
          .hierarchy(data)
          .sum((d) => d.value)
          .sort(
            (a, b) => b.height - a.height || b.value - a.value
          );
        return d3$$1.partition().size([
          height,
          ((root.height + 1) * width) / 3,
        ])(root);
      }

      function render(data) {
        const color = d3$$1.scaleOrdinal(
          d3$$1.quantize(d3$$1.interpolateRainbow, data.children.length + 1)
        );

        const root = iciclePartition(data);
        let focus = root;

        const svg = d3$$1.select('body')
          .append('svg')
          .attr('viewBox', [0, 0, width, height])
          .style('font', '20px sans-serif');

        const cell = svg
          .selectAll('g')
          .data(root.descendants())
          .join('g')
          .attr('transform', (d) => `translate($${d.y0},$${d.x0})`);

        const rect = cell
          .append('rect')
          .attr('width', (d) => d.y1 - d.y0 - 1)
          .attr('height', (d) => rectHeight(d))
          .attr('fill-opacity', 0.6)
          .attr('fill', (d) => {
            if (!d.depth) return '#ccc';
            while (d.depth > 1) d = d.parent;
            return color(d.data.name);
          })
          .style('cursor', 'pointer')
          .on('click', clicked);

        const text = cell
          .append('text')
          .style('user-select', 'none')
          .attr('pointer-events', 'none')
          .attr('x', 4)
          .attr('y', 19)
          .attr('fill-opacity', (d) => +labelVisible(d));

        text.append('tspan').text((d) => d.data.name);

        const tspan = text
          .append('tspan')
          .attr('fill-opacity', (d) => labelVisible(d) * 0.7)
          .text((d) => ` $${d3$$1.format(".4s")(d.value)} seconds`);

        cell.append('title').text(
          (d) =>
            `$${d
            .ancestors()
            .map((d) => d.data.name)
            .reverse()
            .join('/')}\n$${d3$$1.format("e")(d.value)} seconds`
        );

        function clicked(event, p) {
          focus = focus === p ? (p = p.parent) : p;

          root.each(
            (d) =>
              (d.target = {
                x0: ((d.x0 - p.x0) / (p.x1 - p.x0)) * height,
                x1: ((d.x1 - p.x0) / (p.x1 - p.x0)) * height,
                y0: d.y0 - p.y0,
                y1: d.y1 - p.y0,
              })
          );

          const t = cell
            .transition()
            .duration(750)
            .attr(
              'transform',
              (d) => `translate($${d.target.y0},$${d.target.x0})`
            );

          rect
            .transition(t)
            .attr('height', (d) => rectHeight(d.target));
          text
            .transition(t)
            .attr('fill-opacity', (d) => +labelVisible(d.target));
          tspan
            .transition(t)
            .attr(
              'fill-opacity',
              (d) => labelVisible(d.target) * 0.7
            );
        }
      }

      d3$$1.json("$json_path").then(render);

    }(d3));
</script>
</html>
"""


PORT = 8001

if __name__ == "__main__":
    desc = f"""\
{kah.get_program_name()} reads timers and prepares an interactive webpage with the profiling information."""

    parser = kah.init_argparse(desc)
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Part at which to serve the page.",
    )
    parser.add_argument(
        "--html-path",
        default="index.html",
        help="Where to save the HTML file that displays the data."
        " Existing data will be overwritten.",
    )
    parser.add_argument(
        "--json-path",
        default="data.json",
        help="Where to save the JSON file with the data."
        " Existing data will be overwritten.",
    )
    parser.add_argument(
        "--no-server",
        action="store_true",
        help="Do not start a web server.",
    )
    parser.add_argument(
        "--only-server",
        action="store_true",
        help="Only start the server.",
    )

    args = kah.get_args(parser)

    logger = logging.getLogger(__name__)

    if args.verbose:
        logging.basicConfig(format="%(asctime)s - %(message)s")
        logger.setLevel(logging.DEBUG)

    if not args.only_server:
        with SimDir(
            args.datadir,
            ignore_symlinks=args.ignore_symlinks,
            pickle_file=args.pickle_file,
        ) as sim:
            logger.debug("Prepared SimDir")
            timers = sim.timers.average
            logger.debug("Timers read")

        with open(args.html_path, "w") as file_:
            file_.write(
                Template(html_template).substitute(
                    {"json_path": args.json_path}
                )
            )
            logger.debug(f"{args.html_path} written")

        with open(args.json_path, "w") as file_:
            file_.write(timers.to_json())
            logger.debug(f"{args.json_path} written")

    if not args.no_server:
        with socketserver.TCPServer(
            ("", args.port), http.server.SimpleHTTPRequestHandler
        ) as httpd:
            print(f"Serving at port {args.port}. Terminate with C-c.")
            httpd.serve_forever()
