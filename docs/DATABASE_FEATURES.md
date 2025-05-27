# GraphYML Database Features

This document provides an overview of the database features in GraphYML, explaining how it functions as a full-fledged graph database while maintaining its YAML-based storage format.

## Core Database Features

### 1. Query Language

GraphYML includes a simple but powerful query language for searching and filtering nodes:

```
title contains "inception" AND year > 2010
genres contains "Sci-Fi" OR genres contains "Action"
director = "Christopher Nolan" AND NOT rating < 8.5
```

**Supported Operators:**
- Comparison: `=`, `==`, `!=`, `>`, `>=`, `<`, `<=`
- String: `contains`, `startswith`, `endswith`, `matches` (regex)
- Collection: `in`
- Logical: `AND`, `OR`, `NOT`

**Field Access:**
- Dot notation for nested fields: `metadata.created_by`
- Array access for specific elements

### 2. Transaction Support

GraphYML implements ACID transactions to ensure data integrity:

- **Atomicity**: All operations in a transaction succeed or fail together
- **Consistency**: The database remains in a valid state before and after the transaction
- **Isolation**: Concurrent transactions don't interfere with each other
- **Durability**: Committed changes are permanent

**Transaction Operations:**
- Begin transaction
- Create/update/delete nodes within a transaction
- Commit transaction (save all changes)
- Abort transaction (roll back all changes)

### 3. Indexing

GraphYML supports multiple index types to optimize query performance:

- **Hash Index**: Fast exact-match lookups (e.g., by ID or tag)
- **B-tree Index**: Efficient range queries (e.g., year > 2010)
- **Full-text Index**: Text search within string fields
- **Vector Index**: Similarity search using embeddings

Indexes are automatically updated when nodes are modified and can be rebuilt on demand.

### 4. Authentication & Authorization

GraphYML includes a comprehensive user management system:

- **User Authentication**: Secure password hashing with salt
- **Role-Based Access Control**: Predefined roles (Viewer, Editor, Admin)
- **Permission System**: Fine-grained permissions for operations
- **User Management**: Create, update, delete users and roles

### 5. Backup & Recovery

GraphYML provides tools for data protection:

- **Backup**: Create timestamped backups of the entire database
- **Restore**: Restore from a backup point
- **Transaction Logs**: Record of all operations for recovery
- **Crash Recovery**: Automatic recovery from crashes

## Database Architecture

### Storage Layer

GraphYML uses a hybrid storage approach:

- **YAML Files**: Human-readable storage format for nodes
- **JSON Indexes**: Efficient index storage
- **Transaction Logs**: Record of operations for recovery

The storage is organized into directories:
- `/data`: YAML node files organized by category
- `/indexes`: Index files
- `/transactions`: Transaction logs
- `/auth`: User authentication data

### Query Processing

1. **Parser**: Converts query strings into structured query objects
2. **Optimizer**: Selects the most efficient indexes and execution plan
3. **Executor**: Executes the query against the graph
4. **Result Processor**: Formats and returns the results

### Transaction Management

1. **Transaction Manager**: Coordinates transactions
2. **Transaction Log**: Records operations for recovery
3. **Lock Manager**: Prevents conflicts between transactions
4. **Recovery Manager**: Handles crash recovery

## Using Database Features

### Query Examples

```python
# Simple query
results = db.query("title contains 'matrix' AND year > 1998", user)

# Range query
results = db.query("rating >= 8.0 AND runtime < 120", user)

# Complex query
results = db.query("genres contains 'Sci-Fi' AND director = 'Christopher Nolan' AND NOT year < 2010", user)
```

### Transaction Examples

```python
# Begin transaction
tx = db.begin_transaction(user)

# Create node
db.create_node(node_data, user, tx)

# Update node
db.update_node(node_key, updated_data, user, tx)

# Delete node
db.delete_node(node_key, user, tx)

# Commit transaction
db.commit_transaction(tx.tx_id, user)

# Or abort transaction
db.abort_transaction(tx.tx_id, user)
```

### Index Examples

```python
# Create hash index
db.create_index("tag_index", "tags", IndexType.HASH, user)

# Create B-tree index
db.create_index("year_index", "year", IndexType.BTREE, user)

# Create full-text index
db.create_index("title_index", "title", IndexType.FULLTEXT, user)

# Create vector index
db.create_index("embedding_index", "embedding", IndexType.VECTOR, user)
```

## Performance Considerations

### Optimizing Query Performance

- Use appropriate indexes for your query patterns
- Limit the scope of queries when possible
- Use transactions for batch operations
- Consider the size of your dataset when designing queries

### Scaling Considerations

- GraphYML is designed for small to medium-sized datasets
- For very large datasets, consider:
  - Using more efficient index types
  - Implementing sharding strategies
  - Using a hybrid storage approach with a traditional database

## Future Enhancements

Planned improvements to the database features:

1. **Query Language Enhancements**:
   - Support for aggregation functions
   - Graph traversal operations
   - More complex pattern matching

2. **Performance Optimizations**:
   - Improved indexing algorithms
   - Query caching
   - Parallel query execution

3. **Integration Features**:
   - Export to standard graph formats
   - Import from various data sources
   - API for external access

4. **Advanced Security**:
   - Field-level access control
   - Audit logging
   - Encryption at rest

