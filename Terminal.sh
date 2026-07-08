#!/bin/bash

grep -rn "plot_xY" .
find . -type f -name "*.py" ! -name "Terminal.sh" -exec sed -i '' 's/plot_xY/plot_ribbon/g' {} +
grep -rn "plot_xY" .
