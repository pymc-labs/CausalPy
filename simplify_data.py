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

# Find the cell with data simulation
for i, cell in enumerate(data["cells"]):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        if "Simulate baseline sales: trend + seasonality + noise" in source:
            # Replace the seasonal component with monthly pattern (easier for C(month) to capture)
            old_source = cell["source"]
            new_source = []
            for line in old_source:
                if "season = 10 * np.sin" in line:
                    # Replace sine wave with monthly pattern
                    new_source.append(
                        "# Monthly seasonality: easier for C(month) to capture\n"
                    )
                    new_source.append(
                        "month_effects = {1: 5, 2: 3, 3: 8, 4: 10, 5: 12, 6: 15,  # Higher in summer\n"
                    )
                    new_source.append(
                        "                 7: 18, 8: 16, 9: 10, 10: 5, 11: 2, 12: 0}  # Lower in winter\n"
                    )
                    new_source.append(
                        "season = np.array([month_effects[month] for month in dates.month])\n"
                    )
                else:
                    new_source.append(line)
            cell["source"] = new_source
            print(f"Updated seasonal component in cell {i}")
            break
            # Keep the rest of the cell (DataFrame creation, etc.)
            old_source = cell["source"]
            # Find where DataFrame creation starts
            df_start = None
            for j, line in enumerate(old_source):
                if "# Create DataFrame" in line:
                    df_start = j
                    break
            if df_start:
                new_source.extend(old_source[df_start:])
            else:
                # If we can't find it, just add the DataFrame creation
                new_source.extend(
                    [
                        "\n",
                        "# Create DataFrame\n",
                        "df = pd.DataFrame(\n",
                        "    {\n",
                        '        "sales": sales,\n',
                        '        "t": np.arange(n_weeks),\n',
                        '        "month": dates.month,\n',
                        "    },\n",
                        "    index=dates,\n",
                        ")\n",
                    ]
                )
            cell["source"] = new_source
            print(f"Updated cell {i}")
            break

# Write the notebook back
with open(
    "docs/source/notebooks/its_three_period_pymc.ipynb", "w", encoding="utf-8"
) as f:
    json.dump(data, f, indent=1, ensure_ascii=False)

print("Notebook updated successfully!")
