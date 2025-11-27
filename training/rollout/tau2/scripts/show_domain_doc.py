#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# -----------------------------------------------------------------------------

import argparse
import threading
import time
import webbrowser

from loguru import logger

from tau2.environment.server import EnvironmentServer
from tau2.registry import registry

HOST = "127.0.0.1"
PORT = 8004


def open_browser(host: str = HOST, port: int = PORT, delay: float = 1.5):
    """
    Open the browser to the /redoc endpoint after a short delay.

    Args:
        host: Server host address
        port: Server port
        delay: Delay in seconds before opening browser
    """

    def _open():
        time.sleep(delay)  # Give the server time to start
        url = f"http://{host}:{port}/redoc"
        logger.info(f"Opening documentation in browser: {url}")
        webbrowser.open(url)

    thread = threading.Thread(target=_open)
    thread.daemon = True
    thread.start()


def main(domain: str):
    """
    Show documentation for a specific domain.

    Args:
        domain: Name of the domain to show documentation for
    """
    try:
        # Get the environment constructor from registry
        logger.info(f"Setting up environment server for domain: {domain}")
        env_constructor = registry.get_env_constructor(domain)

        # Create the environment
        environment = env_constructor()
        logger.info(f"Created environment for domain: {domain}")

        # Create and start the server
        server = EnvironmentServer(environment)

        open_browser()

        server.run(host=HOST, port=PORT)

    except KeyError:
        available_domains = registry.get_domains()
        logger.error(
            f"Domain '{domain}' not found. Available domains: {available_domains}"
        )
        exit(1)
    except Exception as e:
        logger.error(f"Failed to start domain documentation server: {str(e)}")
        exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show domain documentation")
    parser.add_argument(
        "domain",
        type=str,
        help="Name of the domain to show documentation for (e.g., 'airline', 'mock')",
    )
    args = parser.parse_args()
    main(domain=args.domain)
