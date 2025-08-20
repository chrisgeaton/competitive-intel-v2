"""Authentication and authorization utilities."""

import secrets
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from passlib.context import CryptContext
from jose import JWTError, jwt
from sqlalchemy import select, and_, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from app.models.user import User, UserSession
from app.schemas.auth import TokenData

logger = logging.getLogger(__name__)


def validate_jwt_security() -> tuple[bool, str]:
    """
    Validate JWT security configuration.
    Returns (is_secure: bool, message: str)
    """
    secret_key = settings.SECRET_KEY
    
    # Check key length
    if len(secret_key) < 32:
        return False, "JWT secret key too short (minimum 32 characters)"
    
    # Check for default/insecure keys
    insecure_patterns = [
        "your-secret-key-here",
        "change-this-in-production", 
        "default",
        "secret",
        "password",
        "123456"
    ]
    
    key_lower = secret_key.lower()
    for pattern in insecure_patterns:
        if pattern in key_lower:
            return False, "JWT secret key contains insecure pattern"
    
    # Check for sufficient entropy (basic check)
    unique_chars = len(set(secret_key))
    if unique_chars < 16:
        return False, "JWT secret key lacks sufficient entropy"
    
    return True, "JWT secret key is secure"

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=settings.BCRYPT_ROUNDS
)


class AuthService:
    """Service for handling authentication operations."""
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a plain password against a hashed password."""
        try:
            return pwd_context.verify(plain_password, hashed_password)
        except Exception as e:
            logger.error(f"Error verifying password: {e}")
            return False
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a plain text password."""
        return pwd_context.hash(password)
    
    @staticmethod
    def create_access_token(
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create a JWT access token.
        
        Args:
            data: The data to encode in the token
            expires_delta: Optional custom expiration time
            
        Returns:
            The encoded JWT token
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + settings.access_token_expire_timedelta
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        })
        
        encoded_jwt = jwt.encode(
            to_encode,
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM
        )
        return encoded_jwt
    
    @staticmethod
    def create_refresh_token(
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create a JWT refresh token.
        
        Args:
            data: The data to encode in the token
            expires_delta: Optional custom expiration time
            
        Returns:
            The encoded JWT refresh token
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + settings.refresh_token_expire_timedelta
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh",
            "jti": secrets.token_urlsafe(32)  # Unique token ID
        })
        
        encoded_jwt = jwt.encode(
            to_encode,
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM
        )
        return encoded_jwt
    
    @staticmethod
    def decode_token(token: str) -> Optional[TokenData]:
        """
        Decode and validate a JWT token.
        
        Args:
            token: The JWT token to decode
            
        Returns:
            TokenData if valid, None otherwise
        """
        try:
            payload = jwt.decode(
                token,
                settings.SECRET_KEY,
                algorithms=[settings.ALGORITHM]
            )
            
            # Extract user data from payload
            user_id = payload.get("sub")
            email = payload.get("email")
            session_id = payload.get("session_id")
            
            if user_id is None or email is None:
                return None
            
            return TokenData(
                user_id=int(user_id),
                email=email,
                session_id=session_id,
                exp=payload.get("exp"),
                iat=payload.get("iat"),
                scopes=payload.get("scopes", [])
            )
            
        except JWTError as e:
            logger.error(f"JWT decode error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error decoding token: {e}")
            return None
    
    @staticmethod
    async def authenticate_user(
        db: AsyncSession,
        email: str,
        password: str
    ) -> Optional[User]:
        """
        Authenticate a user by email and password.
        
        Args:
            db: Database session
            email: User's email
            password: User's password
            
        Returns:
            User object if authentication successful, None otherwise
        """
        try:
            # Get user by email
            result = await db.execute(
                select(User).where(User.email == email)
            )
            user = result.scalar_one_or_none()
            
            if not user:
                logger.warning(f"Authentication failed: User not found for email {email}")
                return None
            
            if not user.is_active:
                logger.warning(f"Authentication failed: User {email} is inactive")
                return None
            
            if not user.password_hash:
                logger.warning(f"Authentication failed: User {email} has no password set")
                return None
            
            if not AuthService.verify_password(password, user.password_hash):
                logger.warning(f"Authentication failed: Invalid password for {email}")
                return None
            
            # Update last login
            user.update_last_login()
            await db.commit()
            
            logger.info(f"User {email} authenticated successfully")
            return user
            
        except Exception as e:
            logger.error(f"Error authenticating user {email}: {e}")
            return None
    
    @staticmethod
    async def create_user_session(
        db: AsyncSession,
        user: User,
        remember_me: bool = False
    ) -> UserSession:
        """
        Create a new session for the user.
        
        Args:
            db: Database session
            user: User object
            remember_me: Whether to extend session duration
            
        Returns:
            UserSession object
        """
        try:
            # Clean up old sessions if exceeding limit
            await AuthService.cleanup_user_sessions(db, user.id)
            
            # Generate session token
            session_token = secrets.token_urlsafe(32)
            
            # Calculate expiration
            if remember_me:
                expires_at = datetime.utcnow() + settings.session_remember_me_timedelta
            else:
                expires_at = datetime.utcnow() + settings.session_expire_timedelta
            
            # Create session
            session = UserSession(
                user_id=user.id,
                token=session_token,
                expires_at=expires_at
            )
            
            db.add(session)
            await db.commit()
            await db.refresh(session)
            
            logger.info(f"Created session for user {user.email}")
            return session
            
        except Exception as e:
            logger.error(f"Error creating session for user {user.id}: {e}")
            await db.rollback()
            raise
    
    @staticmethod
    async def cleanup_user_sessions(
        db: AsyncSession,
        user_id: int
    ) -> None:
        """
        Clean up expired and excess sessions for a user.
        
        Args:
            db: Database session
            user_id: User ID
        """
        try:
            # Delete expired sessions
            await db.execute(
                delete(UserSession).where(
                    and_(
                        UserSession.user_id == user_id,
                        UserSession.expires_at < datetime.utcnow()
                    )
                )
            )
            
            # Check session count
            result = await db.execute(
                select(UserSession)
                .where(UserSession.user_id == user_id)
                .order_by(UserSession.created_at.desc())
            )
            sessions = result.scalars().all()
            
            # Delete oldest sessions if exceeding limit
            if len(sessions) >= settings.MAX_SESSIONS_PER_USER:
                sessions_to_delete = sessions[settings.MAX_SESSIONS_PER_USER - 1:]
                for session in sessions_to_delete:
                    await db.delete(session)
            
            await db.commit()
            
        except Exception as e:
            logger.error(f"Error cleaning up sessions for user {user_id}: {e}")
            await db.rollback()
    
    @staticmethod
    async def validate_session(
        db: AsyncSession,
        session_token: str
    ) -> Optional[UserSession]:
        """
        Validate a session token.
        
        Args:
            db: Database session
            session_token: Session token to validate
            
        Returns:
            UserSession if valid, None otherwise
        """
        try:
            result = await db.execute(
                select(UserSession)
                .where(UserSession.token == session_token)
                .options(selectinload(UserSession.user))
            )
            session = result.scalar_one_or_none()
            
            if not session:
                return None
            
            if not session.is_valid:
                # Delete invalid session
                await db.delete(session)
                await db.commit()
                return None
            
            return session
            
        except Exception as e:
            logger.error(f"Error validating session: {e}")
            return None
    
    @staticmethod
    async def revoke_session(
        db: AsyncSession,
        session_token: str
    ) -> bool:
        """
        Revoke (delete) a session.
        
        Args:
            db: Database session
            session_token: Session token to revoke
            
        Returns:
            True if revoked successfully, False otherwise
        """
        try:
            # First find the session
            result = await db.execute(
                select(UserSession).where(UserSession.token == session_token)
            )
            session = result.scalar_one_or_none()
            
            if session:
                await db.delete(session)
                await db.commit()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error revoking session: {e}")
            await db.rollback()
            return False
    
    @staticmethod
    async def revoke_all_user_sessions(
        db: AsyncSession,
        user_id: int,
        except_current: Optional[str] = None
    ) -> int:
        """
        Revoke all sessions for a user.
        
        Args:
            db: Database session
            user_id: User ID
            except_current: Optional current session token to keep
            
        Returns:
            Number of sessions revoked
        """
        try:
            query = delete(UserSession).where(UserSession.user_id == user_id)
            
            if except_current:
                query = query.where(UserSession.token != except_current)
            
            result = await db.execute(query)
            await db.commit()
            
            logger.info(f"Revoked {result.rowcount} sessions for user {user_id}")
            return result.rowcount
            
        except Exception as e:
            logger.error(f"Error revoking all sessions for user {user_id}: {e}")
            await db.rollback()
            return 0


auth_service = AuthService()