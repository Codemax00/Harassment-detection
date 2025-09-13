'use strict';

// Safety Detector Web Interface - Main JavaScript

// Global variables
let currentSession = null;
let websocket = null;
let analysisInProgress = false;

// Global error handler for unexpected errors
window.addEventListener('error', function(event) {
    console.error('Global error caught:', event.error);
    showNotification('An unexpected error occurred. Please refresh the page and try again.', 'error');
    return true;
});

// Global unhandled promise rejection handler
window.addEventListener('unhandledrejection', function(event) {
    console.error('Unhandled promise rejection:', event.reason);
    showNotification('An unexpected error occurred. Please try again.', 'error');
    event.preventDefault();
});

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeInterface();
    setupEventListeners();
    checkSystemStatus();
});

// Initialize interface components
function initializeInterface() {
    // Add fade-in animation to cards
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        setTimeout(() => {
            card.classList.add('fade-in');
        }, index * 100);
    });
    
    // Initialize tooltips if Bootstrap is available
    if (typeof bootstrap !== 'undefined') {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
    
    // Setup auto-dismiss for alerts
    setupAlertAutoDismiss();
}

// Setup event listeners
function setupEventListeners() {
    // Global error handler
    window.addEventListener('error', function(e) {
        console.error('Global error:', e.error);
        showNotification('An unexpected error occurred', 'error');
    });
    
    // Online/offline status
    window.addEventListener('online', function() {
        showNotification('Connection restored', 'success');
        updateConnectionStatus(true);
    });
    
    window.addEventListener('offline', function() {
        showNotification('Connection lost', 'warning');
        updateConnectionStatus(false);
    });
    
    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl+S to save configuration
        if (e.ctrlKey && e.key === 's' && document.getElementById('saveConfigBtn')) {
            e.preventDefault();
            document.getElementById('saveConfigBtn').click();
        }
        
        // Escape to close modals
        if (e.key === 'Escape') {
            const modals = document.querySelectorAll('.modal.show');
            modals.forEach(modal => {
                const modalInstance = bootstrap.Modal.getInstance(modal);
                if (modalInstance) modalInstance.hide();
            });
        }
    });
}

// Check system status
function checkSystemStatus() {
    // Check if we're on a page that needs status updates
    if (window.location.pathname === '/' || window.location.pathname === '/camera') {
        // Simulate status check (in real implementation, this would be an API call)
        setTimeout(() => {
            updateSystemStatus('online');
        }, 1000);
    }
}

// Update system status indicator
function updateSystemStatus(status) {
    const statusElements = document.querySelectorAll('.system-status');
    const statusClass = status === 'online' ? 'text-success' : 'text-danger';
    const statusText = status === 'online' ? 'Online' : 'Offline';
    
    statusElements.forEach(element => {
        element.className = `system-status ${statusClass}`;
        element.textContent = statusText;
    });
}

// Update connection status
function updateConnectionStatus(isOnline) {
    const navbar = document.querySelector('.navbar');
    if (navbar) {
        if (isOnline) {
            navbar.classList.remove('connection-lost');
        } else {
            navbar.classList.add('connection-lost');
        }
    }
}

// Show notification
function showNotification(message, type = 'info', duration = 3000) {
    const alertClass = type === 'error' ? 'danger' : type;
    const iconClass = getIconForType(type);
    
    const alertHTML = `
        <div class="alert alert-${alertClass} alert-dismissible fade show notification-alert" role="alert">
            <i class="fas fa-${iconClass}"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    
    // Find or create notification container
    let container = document.getElementById('notification-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'notification-container';
        container.style.position = 'fixed';
        container.style.top = '80px';
        container.style.right = '20px';
        container.style.zIndex = '9999';
        container.style.maxWidth = '400px';
        document.body.appendChild(container);
    }
    
    // Add notification
    container.insertAdjacentHTML('beforeend', alertHTML);
    
    // Auto-dismiss
    if (duration > 0) {
        setTimeout(() => {
            const alerts = container.querySelectorAll('.notification-alert');
            if (alerts.length > 0) {
                alerts[0].remove();
            }
        }, duration);
    }
}

// Get icon for notification type
function getIconForType(type) {
    const icons = {
        'success': 'check-circle',
        'error': 'exclamation-triangle',
        'warning': 'exclamation-circle',
        'info': 'info-circle'
    };
    return icons[type] || 'info-circle';
}

// Setup auto-dismiss for alerts
function setupAlertAutoDismiss() {
    const alerts = document.querySelectorAll('.alert:not(.alert-dismissible)');
    alerts.forEach(alert => {
        setTimeout(() => {
            alert.style.opacity = '0';
            setTimeout(() => {
                alert.remove();
            }, 300);
        }, 5000);
    });
}

// Format duration
function formatDuration(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Validate IP address
function isValidIP(ip) {
    const ipRegex = /^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$/;
    if (!ipRegex.test(ip)) return false;
    
    const parts = ip.split('.');
    return parts.every(part => {
        const num = parseInt(part, 10);
        return num >= 0 && num <= 255;
    });
}

// Show loading spinner
function showLoading(element, text = 'Loading...') {
    const originalContent = element.innerHTML;
    element.dataset.originalContent = originalContent;
    element.innerHTML = `
        <span class="spinner-border spinner-border-sm me-2" role="status"></span>
        ${text}
    `;
    element.disabled = true;
}

// Hide loading spinner
function hideLoading(element) {
    if (element.dataset.originalContent) {
        element.innerHTML = element.dataset.originalContent;
        delete element.dataset.originalContent;
    }
    element.disabled = false;
}

// Debounce function for search inputs
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Copy text to clipboard
async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        showNotification('Copied to clipboard', 'success', 1500);
    } catch (err) {
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = text;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
        showNotification('Copied to clipboard', 'success', 1500);
    }
}

// Download file
function downloadFile(data, filename, type = 'text/plain') {
    const blob = new Blob([data], { type: type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Form validation helper
function validateForm(formElement) {
    const inputs = formElement.querySelectorAll('input, select, textarea');
    let isValid = true;
    
    inputs.forEach(input => {
        if (input.hasAttribute('required') && !input.value.trim()) {
            input.classList.add('is-invalid');
            isValid = false;
        } else {
            input.classList.remove('is-invalid');
        }
        
        // Custom validation
        if (input.type === 'email' && input.value && !isValidEmail(input.value)) {
            input.classList.add('is-invalid');
            isValid = false;
        }
        
        if (input.dataset.validate === 'ip' && input.value && !isValidIP(input.value)) {
            input.classList.add('is-invalid');
            isValid = false;
        }
    });
    
    return isValid;
}

// Email validation
function isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

// Smooth scroll to element
function scrollToElement(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.scrollIntoView({ behavior: 'smooth' });
    }
}

// Theme management
function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
}

function getTheme() {
    return localStorage.getItem('theme') || 'light';
}

// Initialize theme
document.documentElement.setAttribute('data-theme', getTheme());

// Performance monitoring
function trackPerformance(action, startTime) {
    const endTime = performance.now();
    const duration = endTime - startTime;
    console.log(`Performance: ${action} took ${duration.toFixed(2)}ms`);
    
    // In production, this could send data to analytics
    // analytics.track('performance', { action, duration });
}

// Enhanced fetch with timeout and retry
async function fetchWithTimeout(url, options = {}, timeout = 30000, retries = 3) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);
    
    const fetchOptions = {
        ...options,
        signal: controller.signal
    };
    
    for (let attempt = 1; attempt <= retries; attempt++) {
        try {
            const response = await fetch(url, fetchOptions);
            clearTimeout(timeoutId);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            return response;
        } catch (error) {
            clearTimeout(timeoutId);
            
            if (attempt === retries) {
                if (error.name === 'AbortError') {
                    throw new Error('Request timeout - please check your connection');
                }
                throw error;
            }
            
            // Wait before retry (exponential backoff)
            await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempt) * 1000));
        }
    }
}

// Error boundary for async functions
function withErrorBoundary(asyncFn) {
    return async function(...args) {
        try {
            return await asyncFn.apply(this, args);
        } catch (error) {
            console.error('Error in async function:', error);
            showNotification('An error occurred. Please try again.', 'error');
            throw error;
        }
    };
}

// Local storage helpers
const Storage = {
    set: (key, value) => {
        try {
            localStorage.setItem(key, JSON.stringify(value));
        } catch (e) {
            console.warn('Could not save to localStorage:', e);
        }
    },
    
    get: (key, defaultValue = null) => {
        try {
            const value = localStorage.getItem(key);
            return value ? JSON.parse(value) : defaultValue;
        } catch (e) {
            console.warn('Could not read from localStorage:', e);
            return defaultValue;
        }
    },
    
    remove: (key) => {
        try {
            localStorage.removeItem(key);
        } catch (e) {
            console.warn('Could not remove from localStorage:', e);
        }
    }
};

// Export functions for use in other scripts
window.SafetyDetector = {
    showNotification,
    showLoading,
    hideLoading,
    formatDuration,
    formatFileSize,
    isValidIP,
    validateForm,
    copyToClipboard,
    downloadFile,
    Storage,
    withErrorBoundary,
    fetchWithTimeout
};

// Production-ready initialization message
if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    console.log('%c🛡️ Safety Detector Web Interface', 'color: #007bff; font-size: 16px; font-weight: bold;');
    console.log('%cDevelopment mode - System loaded and ready!', 'color: #28a745; font-size: 12px;');
}
