#!/bin/bash

cp -r data ./BiGGen-Bench-Leaderboard/
cp -r src ./BiGGen-Bench-Leaderboard/
cp requirements.txt ./BiGGen-Bench-Leaderboard/
cp app.py ./BiGGen-Bench-Leaderboard/

cd BiGGen-Bench-Leaderboard
git add .
git commit -m "Update"
git push