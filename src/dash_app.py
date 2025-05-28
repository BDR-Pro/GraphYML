"""
Main Dash application for GraphYML.
Integrates all components and provides the user interface.
"""
import dash
from dash import dcc, html, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import yaml
import json
import os
import base64
import io
import datetime
import numpy as np
import pandas as pd
import networkx as nx
from flask import session

from src.config.settings import load_config, save_config, ensure_directories
from src.models.database import Database
from src.models.auth import Permission, Role
from src.models.indexing import IndexType
from src.visualization.graph_viz import cluster_and_plot, visualize_graph

# Initialize global variables
config = load_config()
config = ensure_directories(config)
database = Database()

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)
server = app.server

# Set up server-side session
server.secret_key = os.urandom(24)

# Helper functions
def is_authenticated():
    """Check if user is authenticated."""
    return session.get('authenticated_user') is not None

def get_current_user():
    """Get the current authenticated user."""
    return session.get('authenticated_user')

def create_navbar():
    """Create the navigation bar."""
    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Logout", href="/logout", id="logout-link")) if is_authenticated() else None,
        ],
        brand="🧠 YAML Graph Knowledge DB",
        brand_href="/",
        color="primary",
        dark=True,
    )

# Layout components
def create_login_layout():
    """Create the login page layout."""
    return dbc.Container([
        html.H2("🔐 Login", className="mt-4"),
        dbc.Row([
            dbc.Col([
                dbc.Input(id="login-username", placeholder="Username", type="text"),
                dbc.Input(id="login-password", placeholder="Password", type="password", className="mt-2"),
                dbc.Button("Login", id="login-button", color="primary", className="mt-3"),
                html.Div(id="login-output")
            ], width=6)
        ])
    ])

def create_settings_layout():
    """Create the settings page layout."""
    return dbc.Container([
        html.H2("⚙️ Settings", className="mt-4"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Save Path"),
                dbc.Input(id="settings-save-path", value=config["save_path"], type="text"),
                dbc.Label("Ollama API URL", className="mt-3"),
                dbc.Input(id="settings-ollama-url", value=config["ollama_url"], type="text"),
            ], width=6),
            dbc.Col([
                dbc.Label("Ollama Model"),
                dbc.Input(id="settings-ollama-model", value=config["ollama_model"], type="text"),
                dbc.Label("Edit Inline", className="mt-3"),
                dbc.Checkbox(id="settings-edit-inline", value=config["edit_inline"]),
            ], width=6)
        ]),
        dbc.Button("Save Settings", id="save-settings-button", color="primary", className="mt-3"),
        html.Div(id="settings-output")
    ])

def create_user_management_layout():
    """Create the user management layout."""
    # Get users if authenticated as admin
    users = []
    user = get_current_user()
    
    if user and user.get('has_permission')(Permission.ADMIN):
        users, error = database.auth_manager.get_users(
            user.get('username'), 
            session.get("admin_password", "")
        )
    
    return dbc.Container([
        html.H2("👥 User Management", className="mt-4"),
        
        # Existing users
        html.H3("Existing Users", className="mt-3"),
        html.Div([
            dbc.Accordion([
                dbc.AccordionItem(
                    [
                        html.P(f"Role: {user_data['role']}"),
                        html.P(f"Permissions: {', '.join(user_data['custom_permissions'])}"),
                        html.P(f"Last Login: {datetime.datetime.fromtimestamp(user_data['last_login']).strftime('%Y-%m-%d %H:%M:%S') if user_data['last_login'] else 'Never'}"),
                        
                        dbc.Label("Change Role"),
                        dcc.Dropdown(
                            id=f"role-dropdown-{user_data['username']}",
                            options=[{"label": r.value, "value": r.value} for r in Role],
                            value=user_data['role'],
                            clearable=False
                        ),
                        dbc.Button(
                            "Update Role", 
                            id=f"update-role-{user_data['username']}", 
                            color="primary", 
                            className="mt-2"
                        ),
                        dbc.Button(
                            "Delete User", 
                            id=f"delete-user-{user_data['username']}", 
                            color="danger", 
                            className="mt-2 ms-2"
                        ),
                    ],
                    title=f"{user_data['username']} ({user_data['role']})"
                )
                for user_data in users
            ], start_collapsed=True, id="users-accordion")
        ]) if users else html.P("No users found or not authorized to view users."),
        
        # Create new user
        html.H3("Create New User", className="mt-4"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Username"),
                dbc.Input(id="new-username", type="text"),
                dbc.Label("Password", className="mt-2"),
                dbc.Input(id="new-password", type="password"),
            ], width=6),
            dbc.Col([
                dbc.Label("Role"),
                dcc.Dropdown(
                    id="new-role",
                    options=[{"label": r.value, "value": r.value} for r in Role],
                    value=Role.USER.value,
                    clearable=False
                ),
            ], width=6)
        ]),
        dbc.Button("Create User", id="create-user-button", color="success", className="mt-3"),
        html.Div(id="user-management-output")
    ])

def create_database_stats_layout():
    """Create the database statistics layout."""
    user = get_current_user()
    
    if not user:
        return html.P("Please login to view statistics.")
    
    stats, error = database.get_statistics(user)
    
    if error:
        return html.P(f"Error: {error}")
    
    return dbc.Container([
        html.H2("📊 Database Statistics", className="mt-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Total Nodes", className="card-title"),
                        html.H3(stats["node_count"], className="card-text")
                    ])
                ])
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Total Links", className="card-title"),
                        html.H3(stats["total_links"], className="card-text")
                    ])
                ])
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Indexes", className="card-title"),
                        html.H3(stats["index_count"], className="card-text")
                    ])
                ])
            ], width=4),
        ], className="mt-3"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("With Embeddings", className="card-title"),
                        html.H3(stats["with_embeddings"], className="card-text")
                    ])
                ])
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Without Embeddings", className="card-title"),
                        html.H3(stats["without_embeddings"], className="card-text")
                    ])
                ])
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Active Transactions", className="card-title"),
                        html.H3(stats["active_transactions"], className="card-text")
                    ])
                ])
            ], width=4),
        ], className="mt-3"),
        
        html.H3("Node Categories", className="mt-4"),
        dcc.Graph(
            id="categories-graph",
            figure=px.bar(
                x=list(stats["categories"].keys()),
                y=list(stats["categories"].values()),
                labels={"x": "Category", "y": "Count"},
                title="Node Count by Category"
            )
        ) if stats["categories"] else html.P("No categories found")
    ])

def create_index_management_layout():
    """Create the index management layout."""
    user = get_current_user()
    
    if not user:
        return html.P("Please login to view indexes.")
    
    indexes, error = database.get_indexes(user)
    
    if error:
        return html.P(f"Error: {error}")
    
    return dbc.Container([
        html.H2("🔍 Index Management", className="mt-4"),
        
        # Existing indexes
        html.H3("Existing Indexes", className="mt-3"),
        html.Div([
            dbc.Accordion([
                dbc.AccordionItem(
                    [
                        html.P(f"Field: {index['field_path']}"),
                        html.P(f"Type: {index['type']}"),
                        html.P(f"Last Updated: {datetime.datetime.fromtimestamp(index['last_updated']).strftime('%Y-%m-%d %H:%M:%S') if index['last_updated'] else 'Never'}"),
                        
                        dbc.Button(
                            "Drop Index", 
                            id=f"drop-index-{index['name']}", 
                            color="danger", 
                            className="mt-2"
                        ) if user.get('has_permission')(Permission.ADMIN) else None,
                    ],
                    title=f"{index['name']} ({index['type']})"
                )
                for index in indexes
            ], start_collapsed=True, id="indexes-accordion")
        ]) if indexes else html.P("No indexes found."),
        
        # Create new index
        html.H3("Create New Index", className="mt-4"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Index Name"),
                dbc.Input(id="new-index-name", type="text"),
                dbc.Label("Field Path", className="mt-2"),
                dbc.Input(id="new-index-field", type="text"),
            ], width=6),
            dbc.Col([
                dbc.Label("Index Type"),
                dcc.Dropdown(
                    id="new-index-type",
                    options=[{"label": t.value, "value": t.value} for t in IndexType],
                    value=IndexType.HASH.value,
                    clearable=False
                ),
            ], width=6)
        ]),
        dbc.Button("Create Index", id="create-index-button", color="success", className="mt-3"),
        html.Div(id="index-management-output")
    ])

def create_node_editor_layout():
    """Create the node editor layout."""
    user = get_current_user()
    
    if not user:
        return html.P("Please login to edit nodes.")
    
    # Get all node IDs
    node_ids = list(database.graph.keys()) if hasattr(database, 'graph') else []
    
    return dbc.Container([
        html.H2("📝 Node Editor", className="mt-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Label("Select Node"),
                dcc.Dropdown(
                    id="node-selector",
                    options=[{"label": node_id, "value": node_id} for node_id in node_ids],
                    placeholder="Select a node to edit"
                ),
            ], width=12)
        ]),
        
        html.Div(id="node-editor-content"),
        
        html.Div(id="node-editor-output")
    ])

def create_node_creator_layout():
    """Create the node creator layout."""
    user = get_current_user()
    
    if not user:
        return html.P("Please login to create nodes.")
    
    return dbc.Container([
        html.H2("➕ Create New Node", className="mt-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Label("Node ID"),
                dbc.Input(id="new-node-id", type="text", placeholder="Enter a unique ID"),
                
                dbc.Label("Title", className="mt-2"),
                dbc.Input(id="new-node-title", type="text"),
                
                dbc.Label("Content", className="mt-2"),
                dbc.Textarea(id="new-node-content", style={"height": "200px"}),
                
                dbc.Label("Tags (comma separated)", className="mt-2"),
                dbc.Input(id="new-node-tags", type="text"),
                
                dbc.Button("Create Node", id="create-node-button", color="success", className="mt-3"),
            ], width=12)
        ]),
        
        html.Div(id="node-creator-output")
    ])

def create_query_interface_layout():
    """Create the query interface layout."""
    user = get_current_user()
    
    if not user:
        return html.P("Please login to query the database.")
    
    return dbc.Container([
        html.H2("🔍 Query Interface", className="mt-4"),
        
        dbc.Tabs([
            dbc.Tab([
                html.P("Search for nodes based on criteria, text, and similarity.", className="mt-3"),
                
                html.H4("Criteria Search", className="mt-3"),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Field"),
                        dbc.Input(id="criteria-field", type="text"),
                    ], width=4),
                    dbc.Col([
                        dbc.Label("Operator"),
                        dcc.Dropdown(
                            id="criteria-operator",
                            options=[
                                {"label": "Equals", "value": "eq"},
                                {"label": "Not Equals", "value": "ne"},
                                {"label": "Greater Than", "value": "gt"},
                                {"label": "Less Than", "value": "lt"},
                                {"label": "Contains", "value": "contains"}
                            ],
                            value="eq"
                        ),
                    ], width=4),
                    dbc.Col([
                        dbc.Label("Value"),
                        dbc.Input(id="criteria-value", type="text"),
                    ], width=4),
                ]),
                dbc.Button("Add Criteria", id="add-criteria-button", color="primary", className="mt-2"),
                
                html.Div(id="criteria-list", className="mt-2"),
                
                html.H4("Text Search", className="mt-3"),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Search Text"),
                        dbc.Input(id="search-text", type="text"),
                    ], width=6),
                    dbc.Col([
                        dbc.Label("Search Fields"),
                        dcc.Dropdown(
                            id="search-fields",
                            options=[
                                {"label": "Title", "value": "title"},
                                {"label": "Overview", "value": "overview"},
                                {"label": "Tagline", "value": "tagline"},
                                {"label": "Description", "value": "description"},
                                {"label": "Summary", "value": "summary"}
                            ],
                            value=["title", "overview"],
                            multi=True
                        ),
                    ], width=6),
                ]),
                
                html.H4("Similarity Search", className="mt-3"),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Similarity Type"),
                        dcc.RadioItems(
                            id="similarity-type",
                            options=[
                                {"label": "None", "value": "none"},
                                {"label": "Text", "value": "text"},
                                {"label": "Node ID", "value": "node_id"}
                            ],
                            value="none",
                            inline=True
                        ),
                    ], width=12),
                ]),
                
                html.Div(id="similarity-options"),
                
                dbc.Button("Execute Query", id="execute-query-button", color="success", className="mt-3"),
                
                html.Div(id="query-results", className="mt-4")
            ], label="Combined Query"),
            
            dbc.Tab([
                html.P("Search for text across multiple fields.", className="mt-3"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Search Text"),
                        dbc.Input(id="text-search-input", type="text"),
                        dbc.Button("Search", id="text-search-button", color="primary", className="mt-2"),
                    ], width=12),
                ]),
                
                html.Div(id="text-search-results", className="mt-4")
            ], label="Text Search"),
            
            dbc.Tab([
                html.P("Search for nodes with similar embeddings.", className="mt-3"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Text for Similarity Search"),
                        dbc.Input(id="embedding-search-input", type="text"),
                    ], width=12),
                    dbc.Col([
                        dbc.Label("Similarity Threshold"),
                        dcc.Slider(
                            id="similarity-threshold",
                            min=0,
                            max=1,
                            step=0.05,
                            value=0.7,
                            marks={i/10: str(i/10) for i in range(0, 11)}
                        ),
                    ], width=6),
                    dbc.Col([
                        dbc.Label("Result Limit"),
                        dcc.Slider(
                            id="result-limit",
                            min=1,
                            max=50,
                            step=1,
                            value=10,
                            marks={i: str(i) for i in [1, 10, 20, 30, 40, 50]}
                        ),
                    ], width=6),
                ]),
                
                dbc.Button("Search by Similarity", id="embedding-search-button", color="primary", className="mt-3"),
                
                html.Div(id="embedding-search-results", className="mt-4")
            ], label="Embedding Similarity"),
        ]),
    ])

def create_visualization_layout():
    """Create the visualization layout."""
    user = get_current_user()
    
    if not user:
        return html.P("Please login to view visualizations.")
    
    if not user.get('has_permission')(Permission.READ):
        return html.P("You don't have permission to view visualizations.")
    
    return dbc.Container([
        html.H2("📊 Visualization", className="mt-4"),
        
        dbc.RadioItems(
            id="viz-type",
            options=[
                {"label": "Clustering", "value": "clustering"},
                {"label": "Interactive Network", "value": "network"}
            ],
            value="clustering",
            inline=True,
            className="mb-3"
        ),
        
        dbc.Button("Generate Visualization", id="generate-viz-button", color="primary"),
        
        html.Div(id="viz-output", className="mt-4")
    ])

def create_backup_restore_layout():
    """Create the backup and restore layout."""
    user = get_current_user()
    
    if not user or not user.get('has_permission')(Permission.ADMIN):
        return html.P("You don't have permission to access backup features.")
    
    return dbc.Container([
        html.H2("💾 Backup & Restore", className="mt-4"),
        
        dbc.Tabs([
            dbc.Tab([
                html.P("Create a backup of the current database.", className="mt-3"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Backup Name"),
                        dbc.Input(id="backup-name", type="text", placeholder="Optional name for the backup"),
                    ], width=12),
                ]),
                
                dbc.Button("Create Backup", id="create-backup-button", color="primary", className="mt-3"),
                
                html.Div(id="backup-output")
            ], label="Create Backup"),
            
            dbc.Tab([
                html.P("Restore from a previous backup.", className="mt-3"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Select Backup"),
                        dcc.Dropdown(
                            id="backup-selector",
                            options=[],  # Will be populated dynamically
                            placeholder="Select a backup to restore"
                        ),
                    ], width=12),
                ]),
                
                dbc.Button("Refresh Backups", id="refresh-backups-button", color="secondary", className="mt-2"),
                dbc.Button("Restore Backup", id="restore-backup-button", color="warning", className="mt-2 ms-2"),
                
                html.Div(id="restore-output")
            ], label="Restore Backup"),
        ]),
    ])

# Main layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    create_navbar(),
    html.Div(id='page-content')
])

# Callbacks
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    """Route to the appropriate page based on URL."""
    if pathname == '/logout':
        # Clear session
        session.pop('authenticated_user', None)
        # Redirect to login
        return create_login_layout()
    
    if not is_authenticated():
        return create_login_layout()
    
    if pathname == '/settings':
        return create_settings_layout()
    elif pathname == '/users':
        return create_user_management_layout()
    elif pathname == '/indexes':
        return create_index_management_layout()
    elif pathname == '/stats':
        return create_database_stats_layout()
    elif pathname == '/backup':
        return create_backup_restore_layout()
    elif pathname == '/query':
        return create_query_interface_layout()
    elif pathname == '/visualization':
        return create_visualization_layout()
    elif pathname == '/create':
        return create_node_creator_layout()
    elif pathname == '/edit':
        return create_node_editor_layout()
    else:
        # Main dashboard
        return dbc.Container([
            html.H1("GraphYML Dashboard", className="mt-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Node Editor", className="card-title"),
                            html.P("Edit existing nodes in the graph database."),
                            dbc.Button("Go to Editor", href="/edit", color="primary")
                        ])
                    ])
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Create Node", className="card-title"),
                            html.P("Create new nodes in the graph database."),
                            dbc.Button("Create Node", href="/create", color="success")
                        ])
                    ])
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Query Database", className="card-title"),
                            html.P("Search and query the graph database."),
                            dbc.Button("Query", href="/query", color="info")
                        ])
                    ])
                ], width=4),
            ], className="mt-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Visualization", className="card-title"),
                            html.P("Visualize the graph database."),
                            dbc.Button("Visualize", href="/visualization", color="secondary")
                        ])
                    ])
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Database Stats", className="card-title"),
                            html.P("View statistics about the database."),
                            dbc.Button("View Stats", href="/stats", color="dark")
                        ])
                    ])
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Management", className="card-title"),
                            html.P("Manage users, indexes, and backups."),
                            dbc.Button("Management", href="/users", color="danger")
                        ])
                    ])
                ], width=4),
            ], className="mt-4"),
        ])

# Login callback
@app.callback(
    Output('login-output', 'children'),
    Input('login-button', 'n_clicks'),
    State('login-username', 'value'),
    State('login-password', 'value'),
    prevent_initial_call=True
)
def login(n_clicks, username, password):
    """Handle login."""
    if not username or not password:
        return html.Div("Username and password are required", className="text-danger mt-2")
    
    user = database.authenticate(username, password)
    
    if user:
        # Store user in session
        session['authenticated_user'] = user
        return dcc.Location(pathname='/', id='redirect-to-home')
    else:
        return html.Div("Invalid username or password", className="text-danger mt-2")

# Settings callback
@app.callback(
    Output('settings-output', 'children'),
    Input('save-settings-button', 'n_clicks'),
    State('settings-save-path', 'value'),
    State('settings-ollama-url', 'value'),
    State('settings-ollama-model', 'value'),
    State('settings-edit-inline', 'checked'),
    prevent_initial_call=True
)
def save_settings(n_clicks, save_path, ollama_url, ollama_model, edit_inline):
    """Save settings."""
    global config, database
    
    config["save_path"] = save_path
    config["ollama_url"] = ollama_url
    config["ollama_model"] = ollama_model
    config["edit_inline"] = edit_inline
    
    save_config(config)
    
    # Reinitialize database with new settings
    database = Database()
    
    return html.Div("Settings saved successfully!", className="text-success mt-2")

# Add more callbacks for other functionality...

# Visualization callback
@app.callback(
    Output('viz-output', 'children'),
    Input('generate-viz-button', 'n_clicks'),
    State('viz-type', 'value'),
    prevent_initial_call=True
)
def generate_visualization(n_clicks, viz_type):
    """Generate visualization."""
    if viz_type == "clustering":
        fig, success = cluster_and_plot(database.graph, config)
        
        if success and fig:
            # Convert matplotlib figure to plotly figure
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode("utf-8")
            
            return html.Img(src=f"data:image/png;base64,{img_str}", style={"width": "100%"})
        else:
            return html.Div("Not enough embeddings to cluster", className="text-warning")
    else:  # Interactive Network
        html_content = visualize_graph(database.graph)
        
        return html.Iframe(
            srcDoc=html_content,
            style={"width": "100%", "height": "600px", "border": "none"}
        )

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

