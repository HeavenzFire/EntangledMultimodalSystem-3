import * as THREE from 'three';

export class HRVRing {
    private scene: THREE.Scene;
    private camera: THREE.PerspectiveCamera;
    private renderer: THREE.WebGLRenderer;
    private ring: THREE.Mesh;
    private material: THREE.MeshBasicMaterial;
    private coherence: number = 0.5;
    private hrvRmssd: number = 0.5;

    constructor(canvas: HTMLCanvasElement) {
        // Initialize Three.js scene
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, canvas.width / canvas.height, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        this.renderer.setSize(canvas.width, canvas.height);

        // Position camera
        this.camera.position.z = 5;

        // Create HRV ring geometry
        const geometry = new THREE.RingGeometry(1, 1.2, 64);
        this.material = new THREE.MeshBasicMaterial({
            color: 0xff0000,
            transparent: true,
            opacity: 0.8,
            side: THREE.DoubleSide
        });
        this.ring = new THREE.Mesh(geometry, this.material);
        this.scene.add(this.ring);

        // Add ambient light
        const ambientLight = new THREE.AmbientLight(0x404040);
        this.scene.add(ambientLight);

        this.animate();
    }

    updateHRV(rmssd: number, coherence: number): void {
        this.hrvRmssd = rmssd;
        this.coherence = coherence;

        // Update ring geometry based on HRV RMSSD
        const innerRadius = 1;
        const outerRadius = 1 + this.hrvRmssd * 0.5;
        const newGeometry = new THREE.RingGeometry(innerRadius, outerRadius, 64);
        this.ring.geometry.dispose();
        this.ring.geometry = newGeometry;

        // Update material color and opacity based on coherence
        const hue = this.coherence; // 0 = red, 1 = green
        this.material.color.setHSL(hue, 1, 0.5);
        this.material.opacity = 0.5 + this.coherence * 0.5;

        // Add glow effect by scaling
        const scale = 1 + this.coherence * 0.2;
        this.ring.scale.setScalar(scale);
    }

    private animate = (): void => {
        requestAnimationFrame(this.animate);

        // Rotate the ring slowly
        this.ring.rotation.z += 0.01;

        this.renderer.render(this.scene, this.camera);
    };

    resize(width: number, height: number): void {
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    dispose(): void {
        this.renderer.dispose();
        this.scene.clear();
    }
}