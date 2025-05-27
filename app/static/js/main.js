class CameraManager {
    constructor() {
        this.cameras = {};
        this.initializeCameras();
        this.startStatusUpdates();
    }

    async initializeCameras() {
        try {
            const response = await fetch('/status');
            const data = await response.json();
            const grid = document.getElementById('camera-grid');
            
            Object.entries(data.cameras).forEach(([name, info]) => {
                if (name !== "CAMERA-NAME") {
                    this.cameras[name] = info;
                    grid.appendChild(this.createCameraCard(name));
                }
            });
            
            this.updateAllStatuses();
        } catch (error) {
            console.error('Initialization error:', error);
            this.showError('Failed to initialize cameras');
        }
    }

    createCameraCard(name) {
        const card = document.createElement('div');
        card.className = 'camera-card';
        card.id = `camera-${name}`;
        
        card.innerHTML = `
            <div class="camera-header">
                <h3 class="text-lg font-semibold text-gray-800">${name}</h3>
                <span class="status-badge status-stopped" id="status-${name}">Stopped</span>
            </div>
            <div class="relative">
                <img src="" class="camera-feed hidden" id="feed-${name}">
                <div class="absolute inset-0 flex items-center justify-center bg-gray-900 bg-opacity-50" id="loading-${name}">
                    <div class="animate-spin rounded-full h-12 w-12 border-4 border-white"></div>
                </div>
            </div>
            <div class="controls">
                <button onclick="cameraManager.startCamera('${name}')" class="btn btn-start">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"/>
                    </svg>
                    <span>Start</span>
                </button>
                <button onclick="cameraManager.stopCamera('${name}')" class="btn btn-stop">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 10l0 4"/>
                    </svg>
                    <span>Stop</span>
                </button>
            </div>
        `;
        
        return card;
    }

    async startCamera(name) {
        try {
            const response = await fetch(`/start/${name}`);
            const data = await response.json();
            
            if (response.ok) {
                const feed = document.getElementById(`feed-${name}`);
                feed.style.display = 'block';
                feed.src = `/video_feed/${name}?t=${new Date().getTime()}`;
                
                document.getElementById(`status-${name}`).textContent = 'Running';
                document.getElementById(`status-${name}`).className = 'status-badge status-running';
                
                document.getElementById(`loading-${name}`).style.display = 'none';
                
                this.showSuccess(`Camera ${name} started successfully`);
            } else {
                throw new Error(data.detail || 'Failed to start camera');
            }
        } catch (error) {
            console.error('Error:', error);
            this.showError(`Failed to start camera ${name}`);
        }
    }

    async stopCamera(name) {
        try {
            const response = await fetch(`/stop/${name}`);
            const data = await response.json();
            
            if (response.ok) {
                const feed = document.getElementById(`feed-${name}`);
                feed.style.display = 'none';
                
                document.getElementById(`status-${name}`).textContent = 'Stopped';
                document.getElementById(`status-${name}`).className = 'status-badge status-stopped';
                
                document.getElementById(`loading-${name}`).style.display = 'flex';
                
                this.showSuccess(`Camera ${name} stopped successfully`);
            } else {
                throw new Error(data.detail || 'Failed to stop camera');
            }
        } catch (error) {
            console.error('Error:', error);
            this.showError(`Failed to stop camera ${name}`);
        }
    }

    async startAllCameras() {
        try {
            const response = await fetch('/start');
            if (response.ok) {
                Object.keys(this.cameras).forEach(name => {
                    if (name !== "CAMERA-NAME") {
                        this.startCamera(name);
                    }
                });
                this.showSuccess('All cameras started successfully');
            }
        } catch (error) {
            console.error('Error:', error);
            this.showError('Failed to start all cameras');
        }
    }

    async stopAllCameras() {
        try {
            const response = await fetch('/stop');
            if (response.ok) {
                Object.keys(this.cameras).forEach(name => {
                    if (name !== "CAMERA-NAME") {
                        this.stopCamera(name);
                    }
                });
                this.showSuccess('All cameras stopped successfully');
            }
        } catch (error) {
            console.error('Error:', error);
            this.showError('Failed to stop all cameras');
        }
    }

    startStatusUpdates() {
        setInterval(() => this.updateAllStatuses(), 5000);
    }

    async updateAllStatuses() {
        try {
            const response = await fetch('/status');
            const data = await response.json();
            
            document.getElementById('last-update').textContent = 
                `Last Updated: ${new Date().toLocaleTimeString()}`;
            
            Object.entries(data.cameras).forEach(([name, info]) => {
                if (name !== "CAMERA-NAME") {
                    const statusBadge = document.getElementById(`status-${name}`);
                    if (statusBadge) {
                        statusBadge.textContent = info.status;
                        statusBadge.className = `status-badge status-${info.status.toLowerCase()}`;
                    }
                }
            });
        } catch (error) {
            console.error('Status update error:', error);
        }
    }

    showSuccess(message) {
        // Implement toast notification for success
        console.log('Success:', message);
    }

    showError(message) {
        // Implement toast notification for error
        console.error('Error:', message);
    }
}

// Initialize camera manager
const cameraManager = new CameraManager();

// Global functions for HTML buttons
window.startAllCameras = () => cameraManager.startAllCameras();
window.stopAllCameras = () => cameraManager.stopAllCameras();

// Format datetime
function formatDateTime(timestamp) {
    if (!timestamp) return '-';
    const date = new Date(timestamp);
    return date.toLocaleString('vi-VN');
}

// Lấy dữ liệu từ API
async function fetchData(startDate = null, endDate = null) {
    try {
        const url = new URL('/api/sheet-data', window.location.origin);
        if (startDate) url.searchParams.append('start_date', startDate);
        if (endDate) url.searchParams.append('end_date', endDate);
        
        const response = await fetch(url);
        if (!response.ok) throw new Error('Network response was not ok');
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error fetching data:', error);
        showToast('Lỗi khi tải dữ liệu', 'error');
        return { data: [], total: 0 };
    }
}

// Cập nhật bảng
function updateContainerTable(data) {
    const tableBody = document.getElementById('containerTableBody');
    tableBody.innerHTML = '';

    if (!data || data.length === 0) {
        const emptyRow = document.createElement('tr');
        emptyRow.innerHTML = `
            <td colspan="5" class="px-6 py-4 text-center text-gray-500">
                <div class="flex flex-col items-center justify-center">
                    <svg class="w-12 h-12 mb-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4"/>
                    </svg>
                    <span>Không có dữ liệu trong khoảng thời gian này</span>
                </div>
            </td>
        `;
        tableBody.appendChild(emptyRow);
        return;
    }

    data.forEach(item => {
        const row = document.createElement('tr');
        row.className = 'hover:bg-gray-50';
        
        // Chuyển đổi confidence về thang điểm 100%
        const confidence = parseFloat(item.confidence || 0) / 100;
        // Giới hạn trong khoảng 0-100%
        const confidenceNormalized = Math.min(Math.max(confidence, 0), 100);
        const confidenceFormatted = confidenceNormalized.toFixed(2);
        
        row.innerHTML = `
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                ${item.container_code || '-'}
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                ${item['Text nhận diện được'] || '-'}
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                ${formatDateTime(item.Timestamp)}
            </td>
            <td class="px-6 py-4 whitespace-nowrap">
                <div class="flex items-center">
                    <div class="w-full bg-gray-200 rounded h-2.5 mr-2">
                        <div class="bg-blue-600 h-2.5 rounded" 
                             style="width: ${confidenceNormalized}%"></div>
                    </div>
                    <span class="text-sm text-gray-900">${confidenceFormatted}%</span>
                </div>
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                ${item.camera || '-'}
            </td>
        `;
        tableBody.appendChild(row);
    });

    const totalCount = document.getElementById('totalCount');
    if (totalCount) {
        totalCount.textContent = `Tổng số: ${data.length}`;
    }
}

// Lọc dữ liệu theo ngày
async function filterData() {
    const startDate = document.getElementById('startDate').value;
    const endDate = document.getElementById('endDate').value;
    
    if (!startDate || !endDate) {
        showToast('Vui lòng chọn khoảng thời gian', 'warning');
        return;
    }
    
    if (startDate > endDate) {
        showToast('Ngày bắt đầu phải nhỏ hơn ngày kết thúc', 'warning');
        return;
    }
    
    const response = await fetchData(startDate, endDate);
    updateContainerTable(response.data);
}

// Làm mới dữ liệu
async function refreshData() {
    document.getElementById('startDate').value = '';
    document.getElementById('endDate').value = '';
    const response = await fetchData();
    updateContainerTable(response.data);
}

// Xuất Excel
async function exportToExcel() {
    const startDate = document.getElementById('startDate').value;
    const endDate = document.getElementById('endDate').value;
    
    try {
        const url = new URL(`/export/${startDate || 'today'}`, window.location.origin);
        window.location.href = url;
    } catch (error) {
        console.error('Error exporting data:', error);
        showToast('Lỗi khi xuất dữ liệu', 'error');
    }
}

// Toast notification
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `fixed bottom-4 right-4 px-6 py-3 rounded-lg text-white ${
        type === 'error' ? 'bg-red-500' : 
        type === 'warning' ? 'bg-yellow-500' : 
        'bg-blue-500'
    }`;
    toast.textContent = message;
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.remove();
    }, 3000);
}

// Khởi tạo dữ liệu khi trang load
document.addEventListener('DOMContentLoaded', () => {
    // Set ngày mặc định
    const today = new Date().toISOString().split('T')[0];
    document.getElementById('startDate').value = today;
    document.getElementById('endDate').value = today;
    
    // Load dữ liệu
    refreshData();
});

// Hàm cập nhật thống kê
async function updateStats() {
    try {
        const response = await fetch('/api/stats');
        if (!response.ok) throw new Error('Failed to fetch stats');
        const stats = await response.json();
        
        // Safely update elements if they exist
        const totalDetectionsElement = document.getElementById('totalDetections');
        const uptimeElement = document.getElementById('uptime');
        
        if (totalDetectionsElement) {
            totalDetectionsElement.textContent = stats.total_containers_detected || 0;
        }
        
        if (uptimeElement) {
            uptimeElement.textContent = stats.system_uptime || '0:00:00';
        }
        
        // Update camera status if data exists
        if (stats.containers_per_camera) {
            updateCameraStatus(stats.containers_per_camera);
        }
    } catch (error) {
        console.error('Error updating stats:', error);
    }
}

// Hàm cập nhật thông số
async function updateParams() {
    const params = {
        MOTION_THRESHOLD: parseInt(document.getElementById('motionThreshold').value),
        CONFIDENCE_THRESHOLD: parseFloat(document.getElementById('confidenceThreshold').value),
        OCR_CONFIDENCE_THRESHOLD: parseFloat(document.getElementById('ocrConfidenceThreshold').value),
        OCR_COOLDOWN: parseFloat(document.getElementById('ocrCooldown').value)
    };

    try {
        const response = await fetch('/api/params', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(params)
        });
        
        if (response.ok) {
            showToast('Cập nhật thông số thành công', 'success');
        } else {
            showToast('Lỗi khi cập nhật thông số', 'error');
        }
    } catch (error) {
        showToast('Lỗi kết nối server', 'error');
    }
}

// Start updating stats when the page loads
document.addEventListener('DOMContentLoaded', () => {
    updateStats(); // Initial update
    setInterval(updateStats, 5000); // Update every 5 seconds
});

function updateCameraStatus(status) {
    const cameraList = document.getElementById('cameraList');
    if (!cameraList) return; // Exit if camera list container doesn't exist

    Object.entries(status).forEach(([camera, data]) => {
        const cameraElement = document.querySelector(`[data-camera="${camera}"]`);
        if (cameraElement) {
            const statusBadge = cameraElement.querySelector('.status-badge');
            if (statusBadge) {
                statusBadge.textContent = data.status || 'Unknown';
                statusBadge.className = `status-badge ${(data.status || '').toLowerCase()}`;
            }
            
            const detectionCount = cameraElement.querySelector('.detection-count');
            if (detectionCount) {
                detectionCount.textContent = data.total_detections || 0;
            }
        }
    });
}

function initializeRealTimeCharts() {
    const detectionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Container Detections',
                data: []
            }]
        },
        options: {
            responsive: true,
            animation: false
        }
    });
    
    // Cập nhật real-time
    setInterval(() => updateCharts(detectionChart), 5000);
}

function addAlertSystem() {
    const alertSystem = new AlertSystem({
        threshold: 0.8,  // Ngưỡng cảnh báo
        notificationDuration: 5000
    });
}

// Camera management
let cameraStates = {
    'ENTRANCE': false,
    'EXIT': false
};

// Update camera status
function updateCameraStatus(cameraId, status) {
    const statusElement = document.getElementById(`camera${cameraId === 'ENTRANCE' ? '1' : '2'}-status`);
    statusElement.textContent = `Trạng thái: ${status}`;
}

// Start camera
async function startCamera(cameraId) {
    try {
        const response = await fetch(`/start/${cameraId}`);
        const data = await response.json();
        
        if (response.ok) {
            cameraStates[cameraId] = true;
            updateCameraStatus(cameraId, 'Đang chạy');
            
            // Start video feed
            const feedElement = document.getElementById(`camera${cameraId === 'ENTRANCE' ? '1' : '2'}-feed`);
            feedElement.src = `/video_feed/${cameraId}`;
            
            // Start polling for detection results
            startPollingDetections(cameraId);
        }
    } catch (error) {
        console.error(`Error starting camera ${cameraId}:`, error);
        updateCameraStatus(cameraId, 'Lỗi khởi động');
    }
}

// Stop camera
async function stopCamera(cameraId) {
    try {
        const response = await fetch(`/stop/${cameraId}`);
        const data = await response.json();
        
        if (response.ok) {
            cameraStates[cameraId] = false;
            updateCameraStatus(cameraId, 'Đã dừng');
            
            // Stop video feed
            const feedElement = document.getElementById(`camera${cameraId === 'ENTRANCE' ? '1' : '2'}-feed`);
            feedElement.src = '';
        }
    } catch (error) {
        console.error(`Error stopping camera ${cameraId}:`, error);
    }
}

// Capture image
async function captureImage(cameraId) {
    try {
        const response = await fetch(`/capture/${cameraId}`);
        const data = await response.json();
        
        if (response.ok) {
            // Update detection results
            updateDetectionResults(cameraId, data);
            
            // Add to recent detections
            addRecentDetection(cameraId, data);
        }
    } catch (error) {
        console.error(`Error capturing image from camera ${cameraId}:`, error);
    }
}

// Update detection results
function updateDetectionResults(cameraId, data) {
    const cameraNum = cameraId === 'ENTRANCE' ? '1' : '2';
    const isPlate = cameraId === 'ENTRANCE';
    
    // Update detection text
    const textElement = document.getElementById(`camera${cameraNum}-${isPlate ? 'plate' : 'container'}`);
    textElement.textContent = data.code || '-';
    
    // Update confidence
    const confidenceElement = document.getElementById(`camera${cameraNum}-confidence`);
    confidenceElement.textContent = data.confidence ? `${data.confidence}%` : '-';
    
    // Update time
    const timeElement = document.getElementById(`camera${cameraNum}-time`);
    timeElement.textContent = data.timestamp || '-';
}

// Add to recent detections
function addRecentDetection(cameraId, data) {
    const recentDetections = document.getElementById('recent-detections');
    const isPlate = cameraId === 'ENTRANCE';
    
    const detectionElement = document.createElement('div');
    detectionElement.className = 'bg-gray-50 p-4 rounded-lg';
    detectionElement.innerHTML = `
        <div class="flex justify-between items-center mb-2">
            <span class="font-medium">${isPlate ? 'Biển số' : 'Mã container'}:</span>
            <span class="text-blue-600">${data.code}</span>
        </div>
        <div class="flex justify-between items-center mb-2">
            <span class="font-medium">Độ tin cậy:</span>
            <span class="text-green-600">${data.confidence}%</span>
        </div>
        <div class="flex justify-between items-center">
            <span class="font-medium">Thời gian:</span>
            <span class="text-gray-600">${data.timestamp}</span>
        </div>
    `;
    
    // Add to the beginning of the list
    recentDetections.insertBefore(detectionElement, recentDetections.firstChild);
    
    // Keep only the last 10 detections
    while (recentDetections.children.length > 10) {
        recentDetections.removeChild(recentDetections.lastChild);
    }
}

// Start polling for detection results
function startPollingDetections(cameraId) {
    if (!cameraStates[cameraId]) return;
    
    fetch(`/detection_results/${cameraId}`)
        .then(response => response.json())
        .then(data => {
            if (data.hasNewDetection) {
                updateDetectionResults(cameraId, data);
                addRecentDetection(cameraId, data);
            }
        })
        .catch(error => console.error(`Error polling detections for camera ${cameraId}:`, error))
        .finally(() => {
            if (cameraStates[cameraId]) {
                setTimeout(() => startPollingDetections(cameraId), 1000);
            }
        });
}

// Start all cameras
async function startAllCameras() {
    await startCamera('ENTRANCE');
    await startCamera('EXIT');
}

// Stop all cameras
async function stopAllCameras() {
    await stopCamera('ENTRANCE');
    await stopCamera('EXIT');
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Set up event listeners
    document.getElementById('startAll').addEventListener('click', startAllCameras);
    document.getElementById('stopAll').addEventListener('click', stopAllCameras);
    
    // Initialize camera status
    updateCameraStatus('ENTRANCE', 'Đang chờ...');
    updateCameraStatus('EXIT', 'Đang chờ...');
});
