# NVIDIA Chart Web Server

This directory contains the web server configuration for accessing the NVIDIA sentiment chart via `stock.stk.name`.

## Files

- `nvidia_chart_viewer.html` - The main chart visualization page
- `nvidia_score_price_dump.txt` - Data file with sentiment and price data
- `server.py` - Python web server
- `setup_domain.sh` - Script to configure domain mapping
- `README.md` - This file

## Setup Instructions

### 1. Configure Domain Mapping

Run the setup script to add domain mapping to your hosts file:

```bash
./setup_domain.sh
```

This adds `127.0.0.1 stock.stk.name` to your `/etc/hosts` file.

### 2. Start the Web Server

```bash
python3 server.py
```

The server will start on port 8080 and display:
- `Server running at http://0.0.0.0:8080/`
- `Access your chart at: http://localhost:8080/`

### 3. Access the Chart

Open your browser and navigate to:
- `http://stock.stk.name:8080/`

## Features

- **Custom Domain**: Access via `stock.stk.name` instead of localhost
- **CORS Enabled**: Allows external access if needed
- **Auto-routing**: Root path `/` automatically serves the chart
- **Local Data**: Uses embedded data to avoid CORS issues

## Troubleshooting

- If domain doesn't work, check `/etc/hosts` contains: `127.0.0.1 stock.stk.name`
- If server fails to start, check if port 8080 is available
- Press `Ctrl+C` to stop the server

## Data Updates

To update the chart data, modify the embedded `csvData` in `nvidia_chart_viewer.html` or implement dynamic data loading from the CSV file.