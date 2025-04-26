#!/usr/bin/env bash
# build.sh - Script executed by Render during deployment builds

# Exit immediately if a command exits with a non-zero status.
set -o errexit

echo "Starting build..."

# Upgrade pip and install dependencies from requirements.txt
pip install --upgrade pip
pip install -r requirements.txt
echo "Dependencies installed."

# Run collectstatic to gather static files into STATIC_ROOT
echo "Running collectstatic..."
python manage.py collectstatic --no-input
echo "Collectstatic finished."

# Apply any outstanding database migrations
echo "Running database migrations..."
python manage.py migrate --no-input
echo "Migrations finished."

echo "Build finished successfully!"
