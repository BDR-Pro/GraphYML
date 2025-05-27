"""
Main Streamlit application for GraphYML.
Integrates all components and provides the user interface.
"""
import streamlit as st
from pathlib import Path
import yaml
import json

from src.config.settings import load_config, save_config, ensure_directories
from src.models.database import Database
from src.models.auth import Permission, Role
from src.models.indexing import IndexType
from src.visualization.graph_viz import cluster_and_plot, visualize_graph


def init_session_state():
    """Initialize session state variables if they don't exist."""
    if 'config' not in st.session_state:
        st.session_state.config = load_config()
        st.session_state.config = ensure_directories(st.session_state.config)
    
    if 'database' not in st.session_state:
        st.session_state.database = Database()
    
    if 'authenticated_user' not in st.session_state:
        st.session_state.authenticated_user = None
    
    if 'current_transaction' not in st.session_state:
        st.session_state.current_transaction = None
    
    if 'current_folder' not in st.session_state:
        st.session_state.current_folder = None
    
    if 'load_errors' not in st.session_state:
        st.session_state.load_errors = []


def show_login_ui():
    """Display login UI and handle authentication."""
    st.subheader("🔐 Login")
    
    col1, col2 = st.columns(2)
    
    with col1:
        username = st.text_input("Username", key="login_username")
    
    with col2:
        password = st.text_input("Password", type="password", key="login_password")
    
    if st.button("Login"):
        user = st.session_state.database.authenticate(username, password)
        
        if user:
            st.session_state.authenticated_user = user
            st.success(f"Welcome, {username}!")
            st.rerun()
        else:
            st.error("Invalid username or password")


def show_settings_ui():
    """Display and handle settings UI."""
    st.subheader("⚙️ Settings")
    
    config = st.session_state.config
    
    col1, col2 = st.columns(2)
    
    with col1:
        save_path = st.text_input("Save Path", config["save_path"])
        ollama_url = st.text_input("Ollama API URL", config["ollama_url"])
    
    with col2:
        ollama_model = st.text_input("Ollama Model", config["ollama_model"])
        edit_inline = st.checkbox("Edit Inline", config["edit_inline"])
    
    if st.button("Save Settings"):
        config["save_path"] = save_path
        config["ollama_url"] = ollama_url
        config["ollama_model"] = ollama_model
        config["edit_inline"] = edit_inline
        
        save_config(config)
        st.session_state.config = config
        
        # Reinitialize database with new settings
        st.session_state.database = Database()
        
        st.success("Settings saved!")


def show_user_management():
    """Display user management UI."""
    st.subheader("👥 User Management")
    
    user = st.session_state.authenticated_user
    
    if not user.has_permission(Permission.ADMIN):
        st.warning("You don't have permission to manage users")
        return
    
    # Get users
    users, error = st.session_state.database.auth_manager.get_users(
        user.username, 
        st.session_state.get("admin_password", "")
    )
    
    if error:
        st.error(error)
        
        # Ask for admin password
        admin_password = st.text_input(
            "Admin Password", 
            type="password",
            key="admin_password"
        )
        
        if st.button("Authenticate"):
            users, error = st.session_state.database.auth_manager.get_users(
                user.username, 
                admin_password
            )
            
            if error:
                st.error(error)
            else:
                st.success("Authentication successful")
                st.rerun()
        
        return
    
    # Display users
    st.write("### Existing Users")
    
    for user_data in users:
        with st.expander(f"{user_data['username']} ({user_data['role']})"):
            st.write(f"**Role:** {user_data['role']}")
            st.write(f"**Permissions:** {', '.join(user_data['custom_permissions'])}")
            
            if user_data['last_login']:
                import datetime
                last_login = datetime.datetime.fromtimestamp(user_data['last_login'])
                st.write(f"**Last Login:** {last_login}")
            else:
                st.write("**Last Login:** Never")
            
            # Role update
            new_role = st.selectbox(
                "Change Role",
                [r.value for r in Role],
                index=[r.value for r in Role].index(user_data['role']),
                key=f"role_{user_data['username']}"
            )
            
            if st.button("Update Role", key=f"update_{user_data['username']}"):
                success = st.session_state.database.auth_manager.update_user_role(
                    user_data['username'],
                    Role(new_role),
                    user.username,
                    st.session_state.get("admin_password", "")
                )
                
                if success:
                    st.success(f"Updated role for {user_data['username']}")
                    st.rerun()
                else:
                    st.error("Failed to update role")
            
            # Delete user
            if st.button("Delete User", key=f"delete_{user_data['username']}"):
                success = st.session_state.database.auth_manager.delete_user(
                    user_data['username'],
                    user.username,
                    st.session_state.get("admin_password", "")
                )
                
                if success:
                    st.success(f"Deleted user {user_data['username']}")
                    st.rerun()
                else:
                    st.error("Failed to delete user")
    
    # Create new user
    st.write("### Create New User")
    
    col1, col2 = st.columns(2)
    
    with col1:
        new_username = st.text_input("Username", key="new_username")
        new_password = st.text_input("Password", type="password", key="new_password")
    
    with col2:
        new_role = st.selectbox(
            "Role",
            [r.value for r in Role],
            index=0,
            key="new_role"
        )
    
    if st.button("Create User"):
        if not new_username or not new_password:
            st.error("Username and password are required")
            return
        
        success = st.session_state.database.auth_manager.create_user(
            new_username,
            new_password,
            Role(new_role)
        )
        
        if success:
            st.success(f"Created user {new_username}")
            st.rerun()
        else:
            st.error("Failed to create user")


def show_database_stats():
    """Display database statistics."""
    st.subheader("📊 Database Statistics")
    
    user = st.session_state.authenticated_user
    
    stats, error = st.session_state.database.get_statistics(user)
    
    if error:
        st.error(error)
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Nodes", stats["node_count"])
        st.metric("Total Links", stats["total_links"])
    
    with col2:
        st.metric("With Embeddings", stats["with_embeddings"])
        st.metric("Without Embeddings", stats["without_embeddings"])
    
    with col3:
        st.metric("Active Transactions", stats["active_transactions"])
        st.metric("Indexes", stats["index_count"])
    
    # Categories
    st.write("### Node Categories")
    
    if stats["categories"]:
        import pandas as pd
        import matplotlib.pyplot as plt
        
        # Create DataFrame
        df = pd.DataFrame({
            "Category": list(stats["categories"].keys()),
            "Count": list(stats["categories"].values())
        })
        
        # Sort by count
        df = df.sort_values("Count", ascending=False)
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(df["Category"], df["Count"])
        ax.set_xlabel("Category")
        ax.set_ylabel("Count")
        ax.set_title("Node Count by Category")
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha="right")
        
        # Adjust layout
        plt.tight_layout()
        
        st.pyplot(fig)
    else:
        st.info("No categories found")


def show_index_management():
    """Display index management UI."""
    st.subheader("🔍 Index Management")
    
    user = st.session_state.authenticated_user
    
    # Get indexes
    indexes, error = st.session_state.database.get_indexes(user)
    
    if error:
        st.error(error)
        return
    
    # Display indexes
    st.write("### Existing Indexes")
    
    if not indexes:
        st.info("No indexes found")
    else:
        for index in indexes:
            with st.expander(f"{index['name']} ({index['type']})"):
                st.write(f"**Field:** {index['field_path']}")
                st.write(f"**Type:** {index['type']}")
                
                if index['last_updated']:
                    import datetime
                    last_updated = datetime.datetime.fromtimestamp(index['last_updated'])
                    st.write(f"**Last Updated:** {last_updated}")
                
                if user.has_permission(Permission.ADMIN):
                    if st.button("Drop Index", key=f"drop_{index['name']}"):
                        success, error = st.session_state.database.drop_index(
                            index['name'],
                            user
                        )
                        
                        if success:
                            st.success(f"Dropped index {index['name']}")
                            st.rerun()
                        else:
                            st.error(f"Failed to drop index: {error}")
    
    # Create new index
    if user.has_permission(Permission.ADMIN):
        st.write("### Create New Index")
        
        col1, col2 = st.columns(2)
        
        with col1:
            index_name = st.text_input("Index Name", key="index_name")
            field_path = st.text_input("Field Path", key="field_path")
        
        with col2:
            index_type = st.selectbox(
                "Index Type",
                [t.value for t in IndexType],
                index=0,
                key="index_type"
            )
        
        if st.button("Create Index"):
            if not index_name or not field_path:
                st.error("Index name and field path are required")
                return
            
            success, error = st.session_state.database.create_index(
                index_name,
                field_path,
                IndexType(index_type),
                user
            )
            
            if success:
                st.success(f"Created index {index_name}")
                st.rerun()
            else:
                st.error(f"Failed to create index: {error}")


def show_backup_restore():
    """Display backup and restore UI."""
    st.subheader("💾 Backup & Restore")
    
    user = st.session_state.authenticated_user
    
    if not user.has_permission(Permission.ADMIN):
        st.warning("You don't have permission to perform backup and restore operations")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Backup Database")
        
        backup_dir = st.text_input("Backup Directory", key="backup_dir")
        
        if st.button("Create Backup"):
            if not backup_dir:
                st.error("Backup directory is required")
                return
            
            success, error = st.session_state.database.backup(backup_dir, user)
            
            if success:
                st.success("Backup created successfully")
            else:
                st.error(f"Backup failed: {error}")
    
    with col2:
        st.write("### Restore Database")
        
        restore_dir = st.text_input("Restore Directory", key="restore_dir")
        
        if st.button("Restore Database"):
            if not restore_dir:
                st.error("Restore directory is required")
                return
            
            # Confirm restore
            confirm = st.checkbox("I understand this will overwrite the current database")
            
            if confirm:
                success, error = st.session_state.database.restore(restore_dir, user)
                
                if success:
                    st.success("Database restored successfully")
                    st.rerun()
                else:
                    st.error(f"Restore failed: {error}")
            else:
                st.warning("Please confirm the restore operation")


def show_node_editor():
    """Display node editor UI."""
    st.subheader("📝 Node Editor")
    
    user = st.session_state.authenticated_user
    
    # Get all nodes
    db = st.session_state.database
    
    # Get node keys
    node_keys = list(db.graph.keys())
    
    if not node_keys:
        st.info("No nodes available to edit")
        return
    
    # Select node
    selected = st.selectbox("Select Node", node_keys)
    
    if not selected:
        return
    
    # Get node
    node, error = db.get_node(selected, user)
    
    if error:
        st.error(error)
        return
    
    # Display node
    if node:
        # Convert to YAML for editing
        node_yaml = yaml.dump(node, sort_keys=False)
        
        # Edit node
        edited = st.text_area("Edit YAML", node_yaml, height=300)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save"):
                try:
                    # Parse YAML
                    updated = yaml.safe_load(edited)
                    
                    # Begin transaction
                    tx = db.begin_transaction(user)
                    
                    if tx:
                        # Update node
                        success, error = db.update_node(selected, updated, user, tx)
                        
                        if success:
                            # Commit transaction
                            commit_success = db.commit_transaction(tx.tx_id, user)
                            
                            if commit_success:
                                st.success("Node updated successfully")
                            else:
                                st.error("Failed to commit transaction")
                        else:
                            # Abort transaction
                            db.abort_transaction(tx.tx_id, user)
                            st.error(f"Failed to update node: {error}")
                    else:
                        st.error("Failed to begin transaction")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        with col2:
            if st.button("🗑️ Delete"):
                # Confirm delete
                confirm = st.checkbox("I understand this will permanently delete the node")
                
                if confirm:
                    # Begin transaction
                    tx = db.begin_transaction(user)
                    
                    if tx:
                        # Delete node
                        success, error = db.delete_node(selected, user, tx)
                        
                        if success:
                            # Commit transaction
                            commit_success = db.commit_transaction(tx.tx_id, user)
                            
                            if commit_success:
                                st.success("Node deleted successfully")
                                st.rerun()
                            else:
                                st.error("Failed to commit transaction")
                        else:
                            # Abort transaction
                            db.abort_transaction(tx.tx_id, user)
                            st.error(f"Failed to delete node: {error}")
                    else:
                        st.error("Failed to begin transaction")
                else:
                    st.warning("Please confirm the delete operation")


def show_node_creator():
    """Display node creator UI."""
    st.subheader("➕ Create Node")
    
    user = st.session_state.authenticated_user
    
    if not user.has_permission(Permission.WRITE):
        st.warning("You don't have permission to create nodes")
        return
    
    # Template node
    template = {
        "id": "",
        "title": "",
        "tags": [],
        "category": "default",
        "links": []
    }
    
    # Convert to YAML for editing
    template_yaml = yaml.dump(template, sort_keys=False)
    
    # Edit template
    edited = st.text_area("Edit YAML", template_yaml, height=300)
    
    if st.button("Create Node"):
        try:
            # Parse YAML
            node_data = yaml.safe_load(edited)
            
            # Begin transaction
            tx = st.session_state.database.begin_transaction(user)
            
            if tx:
                # Create node
                success, node_key, error = st.session_state.database.create_node(
                    node_data, user, tx
                )
                
                if success:
                    # Commit transaction
                    commit_success = st.session_state.database.commit_transaction(
                        tx.tx_id, user
                    )
                    
                    if commit_success:
                        st.success(f"Node {node_key} created successfully")
                        
                        # Clear editor
                        st.session_state["node_creator"] = template_yaml
                    else:
                        st.error("Failed to commit transaction")
                else:
                    # Abort transaction
                    st.session_state.database.abort_transaction(tx.tx_id, user)
                    st.error(f"Failed to create node: {error}")
            else:
                st.error("Failed to begin transaction")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")


def show_query_interface():
    """Display query interface."""
    st.subheader("🔍 Query Database")
    
    user = st.session_state.authenticated_user
    
    # Query types
    query_type = st.radio(
        "Query Type",
        ["Simple Query", "ORM Query", "Text Search", "Embedding Similarity"],
        horizontal=True
    )
    
    if query_type == "Simple Query":
        st.write("""
        ### Query Language
        
        Examples:
        - `title contains "inception"`
        - `year > 2010`
        - `genres contains "Sci-Fi" OR genres contains "Action"`
        - `director = "Christopher Nolan" AND NOT rating < 8.5`
        """)
        
        query = st.text_input("Query", key="simple_query")
        
        if st.button("Execute Query"):
            if not query:
                st.warning("Please enter a query")
                return
            
            results, error = st.session_state.database.query(query, user)
            
            if error:
                st.error(error)
                return
            
            st.success(f"Found {len(results)} results")
            
            # Display results
            for key in results:
                node, _ = st.session_state.database.get_node(key, user)
                
                if node:
                    with st.expander(f"{node.get('title', key)}"):
                        # Display key fields
                        st.write(f"**ID:** {node.get('id', 'N/A')}")
                        st.write(f"**Title:** {node.get('title', 'N/A')}")
                        
                        if 'tags' in node:
                            st.write(f"**Tags:** {', '.join(node['tags'])}")
                        
                        if 'category' in node:
                            st.write(f"**Category:** {node['category']}")
                        
                        # Show full node as JSON
                        with st.expander("View Full Node"):
                            st.json(node)
    
    elif query_type == "ORM Query":
        st.write("""
        ### ORM-Style Query
        
        Use a more user-friendly interface to query the database.
        """)
        
        # Create tabs for different ORM query types
        orm_tabs = st.tabs([
            "By Field", "By Genre/Tag", "By Year/Rating", "By Similarity", "Combined"
        ])
        
        with orm_tabs[0]:
            # Query by field
            field = st.selectbox(
                "Field",
                ["title", "director", "category", "id", "tags", "genres", "year", "rating"],
                key="orm_field"
            )
            
            operator = st.selectbox(
                "Operator",
                ["=", "!=", ">", ">=", "<", "<=", "contains", "startswith", "endswith"],
                key="orm_operator"
            )
            
            value = st.text_input("Value", key="orm_value")
            
            if st.button("Search by Field"):
                if not value:
                    st.warning("Please enter a value")
                    return
                
                results = st.session_state.database.orm.find_by_field(
                    field, value, operator, user
                )
                
                st.success(f"Found {len(results)} results")
                
                # Display results
                for node in results:
                    with st.expander(f"{node.get('title', node.get('id', 'Unknown'))}"):
                        # Display key fields
                        st.write(f"**ID:** {node.get('id', 'N/A')}")
                        st.write(f"**Title:** {node.get('title', 'N/A')}")
                        
                        if 'tags' in node:
                            st.write(f"**Tags:** {', '.join(node['tags'])}")
                        
                        # Show full node as JSON
                        with st.expander("View Full Node"):
                            st.json(node)
        
        with orm_tabs[1]:
            # Query by field contains
            field = st.selectbox(
                "Field",
                ["tags", "genres", "keywords", "categories"],
                key="orm_contains_field"
            )
            
            value = st.text_input(
                "Value",
                key="orm_contains_value"
            )
            
            if st.button("Search by Contains"):
                if not value:
                    st.warning("Please enter a value")
                    return
                
                results = st.session_state.database.orm.find_by_field_contains(
                    field, value, user
                )
                
                st.success(f"Found {len(results)} results")
                
                # Display results
                for node in results:
                    with st.expander(f"{node.get('title', node.get('id', 'Unknown'))}"):
                        # Display key fields
                        st.write(f"**ID:** {node.get('id', 'N/A')}")
                        st.write(f"**Title:** {node.get('title', 'N/A')}")
                        
                        if field in node and isinstance(node[field], list):
                            st.write(f"**{field.capitalize()}:** {', '.join(str(x) for x in node[field])}")
                        
                        # Show full node as JSON
                        with st.expander("View Full Node"):
                            st.json(node)
        
        with orm_tabs[2]:
            # Query by field range
            field = st.selectbox(
                "Field",
                ["year", "rating", "runtime", "budget", "revenue"],
                key="orm_range_field"
            )
            
            min_value = st.number_input(
                f"Min {field.capitalize()}", 
                value=0,
                key=f"orm_min_{field}"
            )
            
            max_value = st.number_input(
                f"Max {field.capitalize()}", 
                value=10000 if field == "year" else 10.0 if field == "rating" else 1000,
                key=f"orm_max_{field}"
            )
            
            if st.button("Search by Range"):
                results = st.session_state.database.orm.find_by_field_range(
                    field, min_value, max_value, user
                )
                
                st.success(f"Found {len(results)} results")
                
                # Display results
                for node in results:
                    with st.expander(f"{node.get('title', node.get('id', 'Unknown'))} ({node.get(field, 'N/A')})"):
                        # Display key fields
                        st.write(f"**ID:** {node.get('id', 'N/A')}")
                        st.write(f"**Title:** {node.get('title', 'N/A')}")
                        st.write(f"**{field.capitalize()}:** {node.get(field, 'N/A')}")
                        
                        # Show full node as JSON
                        with st.expander("View Full Node"):
                            st.json(node)
        
        with orm_tabs[3]:
            # Query by similarity
            similarity_type = st.radio(
                "Similarity Type",
                ["Text", "Node ID"],
                horizontal=True,
                key="orm_similarity_type"
            )
            
            if similarity_type == "Text":
                text = st.text_input("Text", key="orm_similarity_text")
                
                threshold = st.slider(
                    "Similarity Threshold", 
                    0.0, 1.0, 0.7, 0.05,
                    key="orm_similarity_threshold"
                )
                
                limit = st.slider(
                    "Result Limit", 
                    1, 50, 10,
                    key="orm_similarity_limit"
                )
                
                if st.button("Search by Text Similarity"):
                    if not text:
                        st.warning("Please enter text")
                        return
                    
                    results = st.session_state.database.orm.find_by_similarity(
                        text=text,
                        threshold=threshold,
                        limit=limit,
                        user=user
                    )
                    
                    st.success(f"Found {len(results)} results")
                    
                    # Display results
                    for node, score in results:
                        with st.expander(f"{node.get('title', node.get('id', 'Unknown'))} (Score: {score:.2f})"):
                            # Display key fields
                            st.write(f"**ID:** {node.get('id', 'N/A')}")
                            st.write(f"**Title:** {node.get('title', 'N/A')}")
                            st.write(f"**Similarity Score:** {score:.4f}")
                            
                            # Show full node as JSON
                            with st.expander("View Full Node"):
                                st.json(node)
            else:
                # Get all node IDs
                node_ids = list(st.session_state.database.graph.keys())
                
                node_id = st.selectbox(
                    "Node ID",
                    node_ids,
                    key="orm_similarity_node_id"
                )
                
                threshold = st.slider(
                    "Similarity Threshold", 
                    0.0, 1.0, 0.7, 0.05,
                    key="orm_similarity_node_threshold"
                )
                
                limit = st.slider(
                    "Result Limit", 
                    1, 50, 10,
                    key="orm_similarity_node_limit"
                )
                
                if st.button("Search by Node Similarity"):
                    if not node_id:
                        st.warning("Please select a node")
                        return
                    
                    results = st.session_state.database.orm.find_by_similarity(
                        node_id=node_id,
                        threshold=threshold,
                        limit=limit,
                        user=user
                    )
                    
                    st.success(f"Found {len(results)} results")
                    
                    # Display results
                    for node, score in results:
                        with st.expander(f"{node.get('title', node.get('id', 'Unknown'))} (Score: {score:.2f})"):
                            # Display key fields
                            st.write(f"**ID:** {node.get('id', 'N/A')}")
                            st.write(f"**Title:** {node.get('title', 'N/A')}")
                            st.write(f"**Similarity Score:** {score:.4f}")
                            
                            # Show full node as JSON
                            with st.expander("View Full Node"):
                                st.json(node)
        
        with orm_tabs[4]:
            # Combined query
            st.write("### Combined Query")
            st.write("Search using multiple criteria")
            
            st.write("#### Field Criteria")
            
            # Add field criteria
            criteria = {}
            
            # Add field 1
            col1, col2, col3 = st.columns(3)
            with col1:
                field1 = st.selectbox(
                    "Field 1",
                    ["", "title", "director", "category", "year", "rating", "tags", "genres"],
                    key="orm_combined_field1"
                )
            with col2:
                op1 = st.selectbox(
                    "Operator 1",
                    ["=", "!=", ">", ">=", "<", "<=", "contains"],
                    key="orm_combined_op1"
                )
            with col3:
                val1 = st.text_input(
                    "Value 1",
                    key="orm_combined_val1"
                )
            
            if field1 and val1:
                if field1 not in criteria:
                    criteria[field1] = {}
                criteria[field1][op1] = val1
            
            # Add field 2
            col1, col2, col3 = st.columns(3)
            with col1:
                field2 = st.selectbox(
                    "Field 2",
                    ["", "title", "director", "category", "year", "rating", "tags", "genres"],
                    key="orm_combined_field2"
                )
            with col2:
                op2 = st.selectbox(
                    "Operator 2",
                    ["=", "!=", ">", ">=", "<", "<=", "contains"],
                    key="orm_combined_op2"
                )
            with col3:
                val2 = st.text_input(
                    "Value 2",
                    key="orm_combined_val2"
                )
            
            if field2 and val2:
                if field2 not in criteria:
                    criteria[field2] = {}
                criteria[field2][op2] = val2
            
            st.write("#### Text Search")
            
            # Text search
            text_search = {}
            
            search_text = st.text_input(
                "Search Text",
                key="orm_combined_search_text"
            )
            
            search_fields = st.multiselect(
                "Search Fields",
                ["title", "overview", "tagline", "description", "summary"],
                ["title", "overview"],
                key="orm_combined_search_fields"
            )
            
            if search_text and search_fields:
                text_search[tuple(search_fields)] = search_text
            
            st.write("#### Similarity Search")
            
            # Similarity search
            similarity = {}
            
            similarity_type = st.radio(
                "Similarity Type",
                ["None", "Text", "Node ID"],
                key="orm_combined_similarity_type"
            )
            
            if similarity_type == "Text":
                similarity_text = st.text_input(
                    "Text",
                    key="orm_combined_similarity_text"
                )
                
                if similarity_text:
                    similarity["text"] = similarity_text
            
            elif similarity_type == "Node ID":
                # Get all node IDs
                node_ids = list(st.session_state.database.graph.keys())
                
                similarity_node = st.selectbox(
                    "Node ID",
                    node_ids,
                    key="orm_combined_similarity_node"
                )
                
                if similarity_node:
                    similarity["node_id"] = similarity_node
            
            if similarity_type != "None":
                similarity["threshold"] = st.slider(
                    "Similarity Threshold", 
                    0.0, 1.0, 0.7, 0.05,
                    key="orm_combined_similarity_threshold"
                )
            
            if st.button("Execute Combined Query"):
                # Execute query
                results = st.session_state.database.orm.find_by_criteria(
                    criteria=criteria if criteria else None,
                    text_search=text_search if text_search else None,
                    similarity=similarity if similarity and similarity_type != "None" else None,
                    user=user
                )
                
                st.success(f"Found {len(results)} results")
                
                # Display results
                for node in results:
                    with st.expander(f"{node.get('title', node.get('id', 'Unknown'))}"):
                        # Display key fields
                        st.write(f"**ID:** {node.get('id', 'N/A')}")
                        st.write(f"**Title:** {node.get('title', 'N/A')}")
                        
                        # Show common fields if they exist
                        for field in ["year", "rating", "director", "category"]:
                            if field in node:
                                st.write(f"**{field.capitalize()}:** {node[field]}")
                        
                        # Show list fields if they exist
                        for field in ["tags", "genres"]:
                            if field in node and isinstance(node[field], list):
                                st.write(f"**{field.capitalize()}:** {', '.join(str(x) for x in node[field])}")
                        
                        # Show full node as JSON
                        with st.expander("View Full Node"):
                            st.json(node)
    
    elif query_type == "Text Search":
        st.write("""
        ### Text Search
        
        Search for text across multiple fields.
        """)
        
        text = st.text_input("Search Text", key="text_search")
        
        if st.button("Search"):
            if not text:
                st.warning("Please enter search text")
                return
            
            # Use ORM for text search
            results = st.session_state.database.orm.find_by_text_search(
                text,
                ["title", "overview", "tagline"],
                user
            )
            
            st.success(f"Found {len(results)} results")
            
            # Display results
            for node in results:
                with st.expander(f"{node.get('title', node.get('id', 'Unknown'))}"):
                    # Display key fields
                    st.write(f"**ID:** {node.get('id', 'N/A')}")
                    st.write(f"**Title:** {node.get('title', 'N/A')}")
                    
                    if 'tags' in node:
                        st.write(f"**Tags:** {', '.join(node['tags'])}")
                    
                    # Show full node as JSON
                    with st.expander("View Full Node"):
                        st.json(node)
    
    else:  # Embedding Similarity
        st.write("""
        ### Embedding Similarity Search
        
        Search for nodes with similar embeddings.
        """)
        
        text = st.text_input("Text for Similarity Search", key="embedding_search")
        threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.7, 0.05)
        limit = st.slider("Result Limit", 1, 50, 10)
        
        if st.button("Search by Similarity"):
            if not text:
                st.warning("Please enter text for similarity search")
                return
            
            # Use ORM for similarity search
            results = st.session_state.database.orm.find_by_similarity(
                text=text,
                threshold=threshold,
                limit=limit,
                user=user
            )
            
            st.success(f"Found {len(results)} results")
            
            # Display results
            for node, score in results:
                with st.expander(f"{node.get('title', node.get('id', 'Unknown'))} (Score: {score:.2f})"):
                    # Display key fields
                    st.write(f"**ID:** {node.get('id', 'N/A')}")
                    st.write(f"**Title:** {node.get('title', 'N/A')}")
                    st.write(f"**Similarity Score:** {score:.4f}")
                    
                    if 'tags' in node:
                        st.write(f"**Tags:** {', '.join(node['tags'])}")
                    
                    # Show full node as JSON
                    with st.expander("View Full Node"):
                        st.json(node)


def show_visualization():
    """Display graph visualization UI."""
    st.subheader("📊 Visualization")
    
    user = st.session_state.authenticated_user
    
    if not user.has_permission(Permission.READ):
        st.warning("You don't have permission to view visualizations")
        return
    
    viz_type = st.radio(
        "Visualization Type",
        ["Clustering", "Interactive Network"],
        horizontal=True
    )
    
    if viz_type == "Clustering":
        if st.button("🧮 Cluster & Visualize"):
            with st.spinner("Clustering and plotting..."):
                fig, success = cluster_and_plot(
                    st.session_state.database.graph, 
                    st.session_state.config
                )
                
                if success and fig:
                    st.pyplot(fig)
                else:
                    st.warning("Not enough embeddings to cluster")
    
    else:  # Interactive Network
        if st.button("🕸️ Generate Interactive Network"):
            with st.spinner("Generating network visualization..."):
                html = visualize_graph(st.session_state.database.graph)
                
                # Display in an iframe
                st.components.v1.html(html, height=600)


def show_logout_button():
    """Display logout button."""
    if st.button("Logout"):
        st.session_state.authenticated_user = None
        st.rerun()


def main():
    """Main entry point for the Streamlit app."""
    st.set_page_config(
        page_title="YAML Graph DB",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧠 YAML Graph Knowledge DB")
    
    # Initialize session state
    init_session_state()
    
    # Check authentication
    if not st.session_state.authenticated_user:
        show_login_ui()
        return
    
    # Sidebar
    with st.sidebar:
        st.write(f"**Logged in as:** {st.session_state.authenticated_user.username}")
        st.write(f"**Role:** {st.session_state.authenticated_user.role.value}")
        
        show_logout_button()
        
        st.divider()
        
        # Admin settings
        if st.session_state.authenticated_user.has_permission(Permission.ADMIN):
            with st.expander("⚙️ Admin Settings"):
                show_settings_ui()
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Editor", 
        "🔍 Query", 
        "📊 Visualization",
        "🔧 Management",
        "ℹ️ Info"
    ])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            show_node_editor()
        
        with col2:
            show_node_creator()
    
    with tab2:
        show_query_interface()
    
    with tab3:
        show_visualization()
    
    with tab4:
        if st.session_state.authenticated_user.has_permission(Permission.ADMIN):
            subtab1, subtab2, subtab3 = st.tabs([
                "👥 Users",
                "🔍 Indexes",
                "💾 Backup"
            ])
            
            with subtab1:
                show_user_management()
            
            with subtab2:
                show_index_management()
            
            with subtab3:
                show_backup_restore()
        else:
            st.warning("You don't have permission to access management features")
    
    with tab5:
        show_database_stats()


if __name__ == "__main__":
    main()
