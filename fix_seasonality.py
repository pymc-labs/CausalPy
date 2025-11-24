#   Copyright 2025 - 2025 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
import json

# Read the notebook
with open(
    "docs/source/notebooks/its_three_period_pymc.ipynb", "r", encoding="utf-8"
) as f:
    data = json.load(f)

# Find and update the seasonal component
for i, cell in enumerate(data["cells"]):
    if cell.get("cell_type") == "code":
        source_lines = cell.get("source", [])
        # Check if this cell has the seasonality line
        for j, line in enumerate(source_lines):
            if "season = 10 * np.sin" in line:
                # Replace the sine wave seasonality with monthly pattern
                # Remove the old line and insert new ones
                new_lines = source_lines[:j]  # Keep everything before
                new_lines.append(
                    "# Monthly seasonality: easier for C(month) to capture\n"
                )
                new_lines.append(
                    "month_effects = {1: 5, 2: 3, 3: 8, 4: 10, 5: 12, 6: 15,  # Higher in summer\n"
                )
                new_lines.append(
                    "                 7: 18, 8: 16, 9: 10, 10: 5, 11: 2, 12: 0}  # Lower in winter\n"
                )
                new_lines.append(
                    "season = np.array([month_effects[month] for month in dates.month])\n"
                )
                new_lines.extend(source_lines[j + 1 :])  # Keep everything after
                cell["source"] = new_lines
                print(f"Updated seasonal component in cell {i}")
                break

# Write the notebook back
with open(
    "docs/source/notebooks/its_three_period_pymc.ipynb", "w", encoding="utf-8"
) as f:
    json.dump(data, f, indent=1, ensure_ascii=False)

print("Notebook updated successfully!")
