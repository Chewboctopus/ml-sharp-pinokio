class SplatViewer {
    constructor() {
        // Initialize viewer with default settings
        this.cameraPosition = { lat: 0, lng: 0, alt: 100 };  
        this.cameraRotation = { pitch: 0, yaw: 0 };  
        this.isDragging = false;  
    }

    initControls() {
        // Setup the event listeners for camera controls
        document.addEventListener('mousedown', this.onMouseDown.bind(this));
        document.addEventListener('mousemove', this.onMouseMove.bind(this));
        document.addEventListener('mouseup', this.onMouseUp.bind(this));
        document.addEventListener('wheel', this.onMouseWheel.bind(this));
    }

    onMouseDown(event) {
        this.isDragging = true;
        this.lastMousePosition = { x: event.clientX, y: event.clientY };
    }

    onMouseMove(event) {
        if (this.isDragging) {
            const deltaX = event.clientX - this.lastMousePosition.x;
            const deltaY = event.clientY - this.lastMousePosition.y;
            this.cameraRotation.yaw += deltaX * 0.1;
            this.cameraRotation.pitch -= deltaY * 0.1;
            this.lastMousePosition = { x: event.clientX, y: event.clientY };
            this.updateCamera();
        }
    }

    onMouseUp() {
        this.isDragging = false;
    }

    onMouseWheel(event) {
        this.cameraPosition.alt -= event.deltaY * 0.05;
        this.updateCamera();
    }

    updateCamera() {
        // Placeholder for camera update logic
        console.log('Camera updated:', this.cameraPosition, this.cameraRotation);
    }
}

// Usage
const viewer = new SplatViewer();
viewer.initControls();