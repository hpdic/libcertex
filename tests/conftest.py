# Copyright 2026 Dongfang Zhao <dzhao@cs.washington.edu>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""A test argument named Index runs the test against every available implementation of the index."""


def pytest_generate_tests(metafunc):
    if "Index" in metafunc.fixturenames:
        from test_exact import IMPLEMENTATIONS

        metafunc.parametrize("Index", list(IMPLEMENTATIONS.values()), ids=list(IMPLEMENTATIONS))
