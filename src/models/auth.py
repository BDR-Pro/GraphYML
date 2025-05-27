"""
Authentication and access control for GraphYML.
Provides user management and permission checking.
"""
import os
import json
import time
import uuid
import hashlib
import threading
from enum import Enum
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path


class Permission(Enum):
    """Permissions for database operations."""
    READ = "read"  # Read nodes
    WRITE = "write"  # Create/update nodes
    DELETE = "delete"  # Delete nodes
    ADMIN = "admin"  # Administrative operations


class Role(Enum):
    """Built-in roles with predefined permissions."""
    VIEWER = "viewer"  # Read-only access
    EDITOR = "editor"  # Read/write access
    ADMIN = "admin"  # Full access


# Role permission mappings
ROLE_PERMISSIONS = {
    Role.VIEWER: {Permission.READ},
    Role.EDITOR: {Permission.READ, Permission.WRITE},
    Role.ADMIN: {Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN}
}


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """
    Hash a password with a salt.
    
    Args:
        password: Password to hash
        salt: Optional salt (generated if not provided)
        
    Returns:
        Tuple[str, str]: (hashed_password, salt)
    """
    if salt is None:
        salt = uuid.uuid4().hex
    
    # Hash password with salt
    hashed = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000  # Number of iterations
    ).hex()
    
    return hashed, salt


def verify_password(password: str, hashed_password: str, salt: str) -> bool:
    """
    Verify a password against a hash.
    
    Args:
        password: Password to verify
        hashed_password: Stored hash
        salt: Salt used for hashing
        
    Returns:
        bool: Whether the password is correct
    """
    calculated_hash, _ = hash_password(password, salt)
    return calculated_hash == hashed_password


class User:
    """Represents a user with authentication and permissions."""
    
    def __init__(
        self,
        username: str,
        hashed_password: str,
        salt: str,
        role: Role = Role.VIEWER,
        custom_permissions: Optional[Set[Permission]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a user.
        
        Args:
            username: Username
            hashed_password: Hashed password
            salt: Salt used for hashing
            role: User role
            custom_permissions: Optional custom permissions
            metadata: Optional user metadata
        """
        self.username = username
        self.hashed_password = hashed_password
        self.salt = salt
        self.role = role
        self.custom_permissions = custom_permissions or set()
        self.metadata = metadata or {}
        self.last_login = None
    
    def has_permission(self, permission: Permission) -> bool:
        """
        Check if the user has a permission.
        
        Args:
            permission: Permission to check
            
        Returns:
            bool: Whether the user has the permission
        """
        # Check role permissions
        if permission in ROLE_PERMISSIONS.get(self.role, set()):
            return True
        
        # Check custom permissions
        return permission in self.custom_permissions
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert user to dictionary for serialization.
        
        Returns:
            Dict[str, Any]: User data
        """
        return {
            "username": self.username,
            "hashed_password": self.hashed_password,
            "salt": self.salt,
            "role": self.role.value,
            "custom_permissions": [p.value for p in self.custom_permissions],
            "metadata": self.metadata,
            "last_login": self.last_login
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """
        Create a user from a dictionary.
        
        Args:
            data: User data
            
        Returns:
            User: User object
        """
        # Convert role string to enum
        role = Role(data["role"])
        
        # Convert permission strings to enums
        custom_permissions = {
            Permission(p) for p in data.get("custom_permissions", [])
        }
        
        return cls(
            username=data["username"],
            hashed_password=data["hashed_password"],
            salt=data["salt"],
            role=role,
            custom_permissions=custom_permissions,
            metadata=data.get("metadata", {})
        )


class AuthManager:
    """Manages user authentication and authorization."""
    
    def __init__(self, auth_dir: str):
        """
        Initialize the authentication manager.
        
        Args:
            auth_dir: Directory to store authentication data
        """
        self.auth_dir = Path(auth_dir)
        self.auth_dir.mkdir(parents=True, exist_ok=True)
        
        # Path to users file
        self.users_path = self.auth_dir / "users.json"
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Users by username
        self.users = {}
        
        # Load users
        self._load_users()
        
        # Create admin user if no users exist
        if not self.users:
            self._create_default_admin()
    
    def _load_users(self):
        """Load users from disk."""
        with self.lock:
            if self.users_path.exists():
                try:
                    with open(self.users_path, "r", encoding="utf-8") as f:
                        users_data = json.load(f)
                    
                    for user_data in users_data:
                        user = User.from_dict(user_data)
                        self.users[user.username] = user
                
                except Exception as e:
                    print(f"Error loading users: {e}")
    
    def _save_users(self):
        """Save users to disk."""
        with self.lock:
            users_data = [user.to_dict() for user in self.users.values()]
            
            with open(self.users_path, "w", encoding="utf-8") as f:
                json.dump(users_data, f, indent=2)
    
    def _create_default_admin(self):
        """Create a default admin user if no users exist."""
        with self.lock:
            # Default admin credentials
            username = "admin"
            password = "admin"  # Should be changed immediately
            
            # Create admin user
            self.create_user(username, password, Role.ADMIN)
            
            print(f"Created default admin user: {username} / {password}")
            print("Please change the password immediately!")
    
    def create_user(
        self,
        username: str,
        password: str,
        role: Role = Role.VIEWER,
        custom_permissions: Optional[Set[Permission]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create a new user.
        
        Args:
            username: Username
            password: Password
            role: User role
            custom_permissions: Optional custom permissions
            metadata: Optional user metadata
            
        Returns:
            bool: Success flag
        """
        with self.lock:
            if username in self.users:
                return False
            
            # Hash password
            hashed_password, salt = hash_password(password)
            
            # Create user
            user = User(
                username=username,
                hashed_password=hashed_password,
                salt=salt,
                role=role,
                custom_permissions=custom_permissions,
                metadata=metadata
            )
            
            # Add to users
            self.users[username] = user
            
            # Save users
            self._save_users()
            
            return True
    
    def authenticate(self, username: str, password: str) -> Optional[User]:
        """
        Authenticate a user.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            Optional[User]: User object if authentication succeeds, None otherwise
        """
        with self.lock:
            user = self.users.get(username)
            
            if user is None:
                return None
            
            if verify_password(password, user.hashed_password, user.salt):
                # Update last login
                user.last_login = time.time()
                self._save_users()
                
                return user
            
            return None
    
    def change_password(
        self,
        username: str,
        current_password: str,
        new_password: str
    ) -> bool:
        """
        Change a user's password.
        
        Args:
            username: Username
            current_password: Current password
            new_password: New password
            
        Returns:
            bool: Success flag
        """
        with self.lock:
            # Authenticate first
            user = self.authenticate(username, current_password)
            
            if user is None:
                return False
            
            # Hash new password
            hashed_password, salt = hash_password(new_password)
            
            # Update user
            user.hashed_password = hashed_password
            user.salt = salt
            
            # Save users
            self._save_users()
            
            return True
    
    def reset_password(
        self,
        username: str,
        new_password: str,
        admin_username: str,
        admin_password: str
    ) -> bool:
        """
        Reset a user's password (admin only).
        
        Args:
            username: Username
            new_password: New password
            admin_username: Admin username
            admin_password: Admin password
            
        Returns:
            bool: Success flag
        """
        with self.lock:
            # Authenticate admin
            admin = self.authenticate(admin_username, admin_password)
            
            if admin is None or not admin.has_permission(Permission.ADMIN):
                return False
            
            # Get user
            user = self.users.get(username)
            
            if user is None:
                return False
            
            # Hash new password
            hashed_password, salt = hash_password(new_password)
            
            # Update user
            user.hashed_password = hashed_password
            user.salt = salt
            
            # Save users
            self._save_users()
            
            return True
    
    def delete_user(
        self,
        username: str,
        admin_username: str,
        admin_password: str
    ) -> bool:
        """
        Delete a user (admin only).
        
        Args:
            username: Username to delete
            admin_username: Admin username
            admin_password: Admin password
            
        Returns:
            bool: Success flag
        """
        with self.lock:
            # Authenticate admin
            admin = self.authenticate(admin_username, admin_password)
            
            if admin is None or not admin.has_permission(Permission.ADMIN):
                return False
            
            # Check if user exists
            if username not in self.users:
                return False
            
            # Don't allow deleting the last admin
            if self.users[username].role == Role.ADMIN:
                # Count admins
                admin_count = sum(
                    1 for user in self.users.values()
                    if user.role == Role.ADMIN
                )
                
                if admin_count <= 1:
                    return False
            
            # Delete user
            del self.users[username]
            
            # Save users
            self._save_users()
            
            return True
    
    def update_user_role(
        self,
        username: str,
        new_role: Role,
        admin_username: str,
        admin_password: str
    ) -> bool:
        """
        Update a user's role (admin only).
        
        Args:
            username: Username
            new_role: New role
            admin_username: Admin username
            admin_password: Admin password
            
        Returns:
            bool: Success flag
        """
        with self.lock:
            # Authenticate admin
            admin = self.authenticate(admin_username, admin_password)
            
            if admin is None or not admin.has_permission(Permission.ADMIN):
                return False
            
            # Get user
            user = self.users.get(username)
            
            if user is None:
                return False
            
            # Don't allow removing the last admin
            if user.role == Role.ADMIN and new_role != Role.ADMIN:
                # Count admins
                admin_count = sum(
                    1 for u in self.users.values()
                    if u.role == Role.ADMIN
                )
                
                if admin_count <= 1:
                    return False
            
            # Update user
            user.role = new_role
            
            # Save users
            self._save_users()
            
            return True
    
    def get_users(
        self,
        admin_username: str,
        admin_password: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get all users (admin only).
        
        Args:
            admin_username: Admin username
            admin_password: Admin password
            
        Returns:
            Optional[List[Dict[str, Any]]]: List of user data or None
        """
        with self.lock:
            # Authenticate admin
            admin = self.authenticate(admin_username, admin_password)
            
            if admin is None or not admin.has_permission(Permission.ADMIN):
                return None
            
            # Return user data (without sensitive fields)
            return [
                {
                    "username": user.username,
                    "role": user.role.value,
                    "custom_permissions": [p.value for p in user.custom_permissions],
                    "metadata": user.metadata,
                    "last_login": user.last_login
                }
                for user in self.users.values()
            ]
    
    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Get user data (without sensitive fields).
        
        Args:
            username: Username
            
        Returns:
            Optional[Dict[str, Any]]: User data or None
        """
        with self.lock:
            user = self.users.get(username)
            
            if user is None:
                return None
            
            return {
                "username": user.username,
                "role": user.role.value,
                "custom_permissions": [p.value for p in user.custom_permissions],
                "metadata": user.metadata,
                "last_login": user.last_login
            }

