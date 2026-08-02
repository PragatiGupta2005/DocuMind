"""
Application constants related to document uploads.
"""
# Supported document formats
ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".txt"
}
# Maximum upload size (20 MB)
MAX_FILE_SIZE = 20 * 1024 * 1024

# Upload directory
UPLOAD_DIRECTORY = "uploads"

# Allowed MIME types
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "text/plain"
}