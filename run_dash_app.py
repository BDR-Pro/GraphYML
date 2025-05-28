"""
Entry point script to run the Dash application.
"""
from src.dash_app import app

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
