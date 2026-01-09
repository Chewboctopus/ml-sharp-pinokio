// 1. Manage Job Selection & Deletion
function selectJob(jobName) {
    // Update visual selection state
    document.querySelectorAll('.job-list-item').forEach(item => {
        item.classList.remove('selected');
    });
    const selectedItem = document.querySelector('[data-job="' + jobName.replace(/'/g, "\\'") + '"]');
    if (selectedItem) {
        selectedItem.classList.add('selected');
    }

    // Find the textbox and set value
    const textboxContainer = document.getElementById('job_selector_input');
    if (textboxContainer) {
        const textbox = textboxContainer.querySelector('textarea') || textboxContainer.querySelector('input');
        if (textbox) {
            textbox.value = jobName;
            textbox.dispatchEvent(new Event('input', { bubbles: true }));
        }
    }
}

// Global variable for deletion context
let pendingDeleteJob = null;
let pendingDeleteFileBtn = null; // Store the button to click after confirmation

function deleteJob(jobName) {
    pendingDeleteJob = jobName;
    pendingDeleteFileBtn = null;
    showDeleteModal('Job', jobName);
}

function showDeleteModal(type, name) {
    let modal = document.getElementById('delete-confirm-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'delete-confirm-modal';
        modal.className = 'delete-modal-overlay';
        modal.innerHTML = `
            <div class="delete-modal-box">
                <div class="delete-modal-icon">🗑️</div>
                <div class="delete-modal-title">Delete ${type}?</div>
                <div class="delete-modal-message" id="delete-modal-message"></div>
                <div class="delete-modal-buttons">
                    <button class="delete-modal-btn cancel" onclick="hideDeleteModal()">Cancel</button>
                    <button class="delete-modal-btn confirm" onclick="confirmDelete()">Delete</button>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
    }

    // Customize message based on type
    let msg = '';
    if (type === 'Job') {
        msg = 'This will permanently delete job "' + name + '" and all its files.';
    } else {
        msg = 'Are you sure you want to delete this file?';
    }

    document.getElementById('delete-modal-message').textContent = msg;
    // Update title logic if needed, but generic "Delete Type?" is okay or we update it:
    modal.querySelector('.delete-modal-title').textContent = `Delete ${type}?`;

    modal.classList.add('visible');
}

function hideDeleteModal() {
    const modal = document.getElementById('delete-confirm-modal');
    if (modal) {
        modal.classList.remove('visible');
    }
    pendingDeleteJob = null;
    pendingDeleteFileBtn = null;
}

function confirmDelete() {
    const modal = document.getElementById('delete-confirm-modal');
    if (modal) modal.classList.remove('visible');

    // Handle Job Deletion
    if (pendingDeleteJob) {
        const deleteContainer = document.getElementById('job_delete_input');
        if (deleteContainer) {
            const textbox = deleteContainer.querySelector('textarea') || deleteContainer.querySelector('input');
            if (textbox) {
                textbox.value = pendingDeleteJob;
                textbox.dispatchEvent(new Event('input', { bubbles: true }));
            }
        }
    }

    // Handle File Deletion
    if (pendingDeleteFileBtn) {
        // Extract filename from the row to send to Python
        const row = pendingDeleteFileBtn.closest('tr') || pendingDeleteFileBtn.closest('.file-preview-item');
        let fileName = '';
        if (row) {
            // Strategy: Iterate TDs to find the one with the filename
            const tds = row.querySelectorAll('td');
            if (tds.length > 0) {
                // It's a table row
                for (let i = 0; i < tds.length; i++) {
                    const td = tds[i];
                    // Skip if it contains the delete button
                    if (td.querySelector('button') || td.querySelector('.remove-button')) continue;
                    // Skip if it seems to be the download/size cell (heuristic)
                    const text = (td.innerText || td.textContent).trim();
                    if (!text) continue;
                    if (td.classList.contains('download')) continue;
                    // Check for size pattern (e.g. "1.2 MB", "500 KB")
                    // Use string check or simple regex 
                    // To avoid escaping hell: check if it ENDS with B and has numbers
                    if (/[\d\.]+\s*(KB|MB|GB|B)$/i.test(text)) continue;

                    // Found a candidate
                    fileName = text;
                    break;
                }
            }

            // Fallback if no specific TD found or not a table
            if (!fileName) {
                // Try to split row text and avoid size
                let fullText = (row.innerText || row.textContent).trim();
                // Safer split: remove CR then split by LF
                let parts = fullText.replace(/\r/g, '').split('\n');
                // Filter out parts that look like size
                for (let p of parts) {
                    p = p.trim();
                    if (p && !/[\d\.]+\s*(KB|MB|GB|B)$/i.test(p) && p !== '✕') {
                        fileName = p;
                        break;
                    }
                }
            }
        }

        if (fileName) {
            const fileInput = document.getElementById('file_delete_input');
            if (fileInput) {
                const textbox = fileInput.querySelector('textarea') || fileInput.querySelector('input');
                if (textbox) {
                    textbox.value = fileName;
                    textbox.dispatchEvent(new Event('input', { bubbles: true }));
                }
            }
        }

        // Also click the UI button to clear it visually immediately (client-side)
        // We need to bypass our own interceptor
        pendingDeleteFileBtn.dataset.confirmed = 'true';
        pendingDeleteFileBtn.click();
    }

    pendingDeleteJob = null;
    pendingDeleteFileBtn = null;
}

// Function called by the Delete Job button via js parameter
function triggerDeleteCurrentJob() {
    const selectorContainer = document.getElementById('job_selector_input');
    if (selectorContainer) {
        const textbox = selectorContainer.querySelector('textarea') || selectorContainer.querySelector('input');
        if (textbox && textbox.value) {
            deleteJob(textbox.value);
        } else {
            alert('No job selected to delete.');
        }
    } else {
        console.error("Job selector input not found");
    }
}

// 2. Manage File List Buttons (Robust)

// Global click listener for file deletion confirmation
// We use capture phase (true) to intercept before Gradio/Svelte
document.addEventListener('click', function (e) {
    // Check if target is a remove button or inside one
    const btn = e.target.closest('button[aria-label="Remove this file"], button.label-clear-button');

    if (btn) {
        // Double check we are inside the file list
        if (document.getElementById('det_files_list') && document.getElementById('det_files_list').contains(btn)) {

            // If already confirmed programmatically, let it pass
            if (btn.dataset.confirmed === 'true') {
                btn.dataset.confirmed = 'false'; // reset
                return; // Allow default Gradio action
            }

            // Otherwise stop it and show modal
            e.preventDefault();
            e.stopPropagation();
            e.stopImmediatePropagation(); // Crucial to stop other listeners

            pendingDeleteFileBtn = btn;
            showDeleteModal('File', 'selected file'); // We could try to extract filename but generic is safer
        }
    }
}, true); // Capture phase!

// Polling loop to hide image delete buttons
// This is more robust than MutationObserver against Svelte re-renders
setInterval(() => {
    const listIds = ['det_files_list', 'new_result_files_list'];

    listIds.forEach(id => {
        const fileList = document.getElementById(id);
        if (!fileList) return;

        // Select ALL buttons and check labels/titles
        const allButtons = fileList.querySelectorAll('button');

        allButtons.forEach(btn => {
            const label = (btn.getAttribute('aria-label') || btn.getAttribute('title') || '').toLowerCase();
            const text = (btn.innerText || btn.textContent || '').trim().toLowerCase();

            const isRemoveBtn = label.includes('remove') || label.includes('clear') || label.includes('delete') ||
                text === '✕' || text === 'x' || btn.classList.contains('label-clear-button');

            if (isRemoveBtn) {
                const row = btn.closest('tr') || btn.closest('.file-preview-item');
                if (row) {
                    const rowText = (row.innerText || row.textContent).toLowerCase();
                    const isImage = rowText.includes('.png') || rowText.includes('.jpg') || rowText.includes('.jpeg') || rowText.includes('.webp');

                    // Hide if it's the New Job list (all files) OR if it's an image in the Detail list
                    const shouldHide = (id === 'new_result_files_list') || isImage;

                    if (shouldHide) {
                        if (btn.style.visibility !== 'hidden') {
                            btn.style.visibility = 'hidden';
                            btn.style.opacity = '0';
                            btn.style.pointerEvents = 'none';
                            btn.style.display = ''; // Keep for layout
                        }
                    } else {
                        // Keep visible for other files in History (like PLYs/MP4s if user wants to delete? 
                        // Actually user said "hide unnecessary file deletion buttons". 
                        // If they have a separate "Delete Job" button, maybe they want them all hidden?
                        // Let's stick to hiding everything in New Job and images in History.
                        if (btn.style.visibility === 'hidden') {
                            btn.style.visibility = 'visible';
                            btn.style.opacity = '1';
                            btn.style.pointerEvents = '';
                        }
                    }
                }
            }
        });
    });
}, 500); // Check every 500ms

// Keep the Event Listeners for explicit actions to trigger immediate Reset
function checkAndResetLibrary() {
    const resetBtn = document.getElementById('btn_lib_reset');
    if (resetBtn) resetBtn.click();
}

document.addEventListener('click', (e) => {
    const closeBtn = e.target.closest('button[aria-label="Close"]');
    if (closeBtn) checkAndResetLibrary();
}, true);

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') checkAndResetLibrary();
});

// Processing Animation Toggle with Timer
// Monitors components and shows pulsing animation + execution timer
(function () {
    let processingActive = false;
    let timerStartTime = null;
    let timerElement = null;
    let timerInterval = null;
    let currentContext = null; // 'new_job' or 'history'

    // Component tracking: which are waiting vs ready
    const componentState = {
        ply: false,    // true when PLY is loaded
        videos: false  // true when videos are loaded
    };

    function createTimerElement() {
        if (!timerElement) {
            timerElement = document.createElement('div');
            timerElement.id = 'execution-timer';
            timerElement.textContent = '00:00';
            document.body.appendChild(timerElement);
        }
        return timerElement;
    }

    function formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }

    function startTimer() {
        timerStartTime = Date.now();
        const timer = createTimerElement();
        timer.classList.add('active');

        if (timerInterval) clearInterval(timerInterval);
        timerInterval = setInterval(() => {
            const elapsed = (Date.now() - timerStartTime) / 1000;
            timer.textContent = formatTime(elapsed);
        }, 100);
    }

    function stopTimer() {
        if (timerInterval) {
            clearInterval(timerInterval);
            timerInterval = null;
        }
        if (timerElement) {
            timerElement.classList.remove('active');
        }
    }

    function setComponentWaiting(elementId, waiting) {
        const el = document.getElementById(elementId);
        if (el) {
            if (waiting) {
                el.classList.add('waiting-component');
            } else {
                el.classList.remove('waiting-component');
            }
        }
    }

    function getLogText(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return '';
        const textarea = container.querySelector('textarea');
        return textarea ? textarea.value : '';
    }

    function checkProcessingState() {
        // Check New Job tab
        const newJobLog = getLogText('new_job_log');
        const detJobLog = getLogText('det_job_log');

        // Determine which context is active
        const newJobProcessing = newJobLog.includes('[Training]') || newJobLog.includes('[50%]') || newJobLog.includes('[70%]');

        // History tab: distinguish between PLY generation and video rendering
        const detPlyProcessing = detJobLog.includes('[Generating]');
        const detVideoProcessing = detJobLog.includes('[Rendering]');
        const detJobProcessing = detPlyProcessing || detVideoProcessing;

        const newJobCompleted = newJobLog.includes('Completed:') || newJobLog.includes('Error:');
        // Simple detection: look for [Done] marker (added at end of all operations)
        const detJobCompleted = detJobLog.includes('[Done]');

        // Debug: log when [Done] is detected
        if (detJobCompleted && processingActive) {
            console.log('[Animation] Detected [Done] marker, stopping animation');
        }

        // New Job context
        if (newJobProcessing && !newJobCompleted && !processingActive) {
            processingActive = true;
            currentContext = 'new_job';
            componentState.ply = false;
            componentState.videos = false;
            startTimer();

            const logContainer = document.getElementById('new_job_log');
            const btnRun = document.getElementById('btn_start_gen');
            if (logContainer) logContainer.classList.add('processing-active');
            if (btnRun) {
                const btn = btnRun.querySelector('button') || btnRun;
                btn.classList.add('btn-processing');
            }

            setComponentWaiting('new_model_3d', true);
            setComponentWaiting('new_vid_color', true);
            setComponentWaiting('new_vid_depth', true);
        }

        // History tab context - PLY generation
        if (detPlyProcessing && !detVideoProcessing && !detJobCompleted && !processingActive) {
            processingActive = true;
            currentContext = 'history_ply';
            componentState.ply = false;
            startTimer();

            const logContainer = document.getElementById('det_job_log');
            if (logContainer) logContainer.classList.add('processing-active');

            // Only illuminate 3D viewer for PLY generation
            setComponentWaiting('det_model_3d', true);
        }

        // History tab context - Video rendering
        if (detVideoProcessing && !detJobCompleted && !processingActive) {
            processingActive = true;
            currentContext = 'history_video';
            componentState.videos = false;
            startTimer();

            const logContainer = document.getElementById('det_job_log');
            if (logContainer) logContainer.classList.add('processing-active');

            // Only illuminate video boxes for video rendering
            setComponentWaiting('det_vid_color', true);
            setComponentWaiting('det_vid_depth', true);
        }

        // Handle progressive unlocking for New Job
        if (currentContext === 'new_job' && processingActive) {
            const plyReady = newJobLog.includes('[50%]') || newJobLog.includes('PLY Generated') || newJobLog.includes('ply_ready') || newJobLog.includes('input_source.ply');
            const videosReady = newJobLog.includes('Completed:');
            const videoRenderingStarted = newJobLog.includes('Engine: Rendering') || newJobLog.includes('[Rendering]');

            // If PLY is ready OR video rendering has started, stop 3D viewer animation
            if ((plyReady || videoRenderingStarted) && !componentState.ply) {
                componentState.ply = true;
                setComponentWaiting('new_model_3d', false);
            }
            if (videosReady && !componentState.videos) {
                componentState.videos = true;
                setComponentWaiting('new_vid_color', false);
                setComponentWaiting('new_vid_depth', false);
            }

            if (newJobCompleted) {
                processingActive = false;
                currentContext = null;
                stopTimer();

                const logContainer = document.getElementById('new_job_log');
                const btnRun = document.getElementById('btn_start_gen');
                if (logContainer) logContainer.classList.remove('processing-active');
                if (btnRun) {
                    const btn = btnRun.querySelector('button') || btnRun;
                    btn.classList.remove('btn-processing');
                }
                setComponentWaiting('new_model_3d', false);
                setComponentWaiting('new_vid_color', false);
                setComponentWaiting('new_vid_depth', false);
            }
        }

        // Handle completion for History PLY
        if (currentContext === 'history_ply' && processingActive && detJobCompleted) {
            processingActive = false;
            currentContext = null;
            stopTimer();

            const logContainer = document.getElementById('det_job_log');
            if (logContainer) logContainer.classList.remove('processing-active');
            setComponentWaiting('det_model_3d', false);
        }

        // Handle completion for History Video
        if (currentContext === 'history_video' && processingActive && detJobCompleted) {
            processingActive = false;
            currentContext = null;
            stopTimer();

            const logContainer = document.getElementById('det_job_log');
            if (logContainer) logContainer.classList.remove('processing-active');
            setComponentWaiting('det_vid_color', false);
            setComponentWaiting('det_vid_depth', false);
        }

        // Fallback: if History processing is active but no marker for 10+ seconds after log stops changing, stop
        if ((currentContext === 'history_ply' || currentContext === 'history_video') && processingActive) {
            if (!window._lastHistoryLogLength) window._lastHistoryLogLength = 0;
            if (!window._historyLogStaleTime) window._historyLogStaleTime = null;

            if (detJobLog.length !== window._lastHistoryLogLength) {
                window._lastHistoryLogLength = detJobLog.length;
                window._historyLogStaleTime = Date.now();
            } else {
                // Check for staleness, but ONLY if document is visible
                // Background tabs get throttled timers, so Date.now() jumps.
                if (document.hidden) {
                    // Reset the clock while hidden so we don't timeout immediately on return
                    window._historyLogStaleTime = Date.now();
                } else if (window._historyLogStaleTime && (Date.now() - window._historyLogStaleTime > 30000)) {
                    // Log hasn't changed for 30 seconds (and we are visible) - assume complete
                    console.log('[Animation] Fallback: log stale for 5s, stopping animation');
                    processingActive = false;
                    const ctx = currentContext;
                    currentContext = null;
                    stopTimer();

                    const logContainer = document.getElementById('det_job_log');
                    if (logContainer) logContainer.classList.remove('processing-active');
                    if (ctx === 'history_ply') setComponentWaiting('det_model_3d', false);
                    if (ctx === 'history_video') {
                        setComponentWaiting('det_vid_color', false);
                        setComponentWaiting('det_vid_depth', false);
                    }
                    window._historyLogStaleTime = null;
                }
            }
        }
    }

    // Check periodically
    setInterval(checkProcessingState, 300);
})();
