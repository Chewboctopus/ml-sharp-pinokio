// 1. Manage Job Selection & Deletion
function selectJob(jobName) {
    if (document.body.classList.contains('ui-locked')) return;

    document.querySelectorAll('.job-list-item').forEach(item => {
        item.classList.remove('selected');
    });
    const selectedItem = document.querySelector('[data-job="' + jobName.replace(/'/g, "\\'") + '"]');
    if (selectedItem) {
        selectedItem.classList.add('selected');
    }

    const textboxContainer = document.getElementById('job_selector_input');
    if (textboxContainer) {
        const textbox = textboxContainer.querySelector('textarea') || textboxContainer.querySelector('input');
        if (textbox) {
            textbox.value = jobName;
            textbox.dispatchEvent(new Event('input', { bubbles: true }));
        }
    }
}

let pendingDeleteJob = null;
let pendingDeleteFileBtn = null;

function deleteJob(jobName) {
    if (document.body.classList.contains('ui-locked')) return;
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

    let msg = type === 'Job' ? 'This will permanently delete job "' + name + '" and all its files.' : 'Are you sure you want to delete this file?';
    document.getElementById('delete-modal-message').textContent = msg;
    modal.querySelector('.delete-modal-title').textContent = `Delete ${type}?`;
    modal.classList.add('visible');
}

function hideDeleteModal() {
    const modal = document.getElementById('delete-confirm-modal');
    if (modal) modal.classList.remove('visible');
    pendingDeleteJob = null;
    pendingDeleteFileBtn = null;
}

function confirmDelete() {
    const modal = document.getElementById('delete-confirm-modal');
    if (modal) modal.classList.remove('visible');

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

    if (pendingDeleteFileBtn) {
        const row = pendingDeleteFileBtn.closest('tr') || pendingDeleteFileBtn.closest('.file-preview-item');
        let fileName = '';
        if (row) {
            const tds = row.querySelectorAll('td');
            if (tds.length > 0) {
                for (let i = 0; i < tds.length; i++) {
                    const td = tds[i];
                    if (td.querySelector('button') || td.querySelector('.remove-button')) continue;
                    const text = (td.innerText || td.textContent).trim();
                    if (!text || td.classList.contains('download') || /[\d\.]+\s*(KB|MB|GB|B)$/i.test(text)) continue;
                    fileName = text;
                    break;
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
        pendingDeleteFileBtn.dataset.confirmed = 'true';
        pendingDeleteFileBtn.click();
    }
    pendingDeleteJob = null;
    pendingDeleteFileBtn = null;
}

function triggerDeleteCurrentJob() {
    const selectorContainer = document.getElementById('job_selector_input');
    if (selectorContainer) {
        const textbox = selectorContainer.querySelector('textarea') || selectorContainer.querySelector('input');
        if (textbox && textbox.value) deleteJob(textbox.value);
        else alert('No job selected to delete.');
    }
}

// Global click interceptor - Works because pointer-events are enabled
document.addEventListener('click', function (e) {
    if (document.body.classList.contains('ui-locked')) {
        // Elements to block
        const isTab = e.target.closest('.tab-nav') || e.target.closest('button[role="tab"]');
        const isHistory = e.target.closest('.job-list-container');
        const isRegenBtn = e.target.closest('#btn_gen_ply_details') || e.target.closest('#btn_gen_video_details');
        const isStartBtn = e.target.closest('#btn_start_gen');

        if (isTab || isHistory || isRegenBtn || isStartBtn) {
            console.log("[UI] Interaction blocked: system busy.");
            e.preventDefault();
            e.stopPropagation();
            e.stopImmediatePropagation();
            return;
        }

        // Block delete file button while busy
        const deleteBtn = e.target.closest('button[aria-label="Remove this file"], button.label-clear-button');
        if (deleteBtn && document.getElementById('det_files_list') && document.getElementById('det_files_list').contains(deleteBtn)) {
            e.preventDefault();
            e.stopPropagation();
            return;
        }
    } else {
        // Standard File delete confirmation logic
        const btn = e.target.closest('button[aria-label="Remove this file"], button.label-clear-button');
        if (btn && document.getElementById('det_files_list') && document.getElementById('det_files_list').contains(btn)) {
            if (btn.dataset.confirmed === 'true') {
                btn.dataset.confirmed = 'false';
                return;
            }
            e.preventDefault();
            e.stopPropagation();
            e.stopImmediatePropagation();
            pendingDeleteFileBtn = btn;
            showDeleteModal('File', 'selected file');
        }
    }
}, true);

setInterval(() => {
    const listIds = ['det_files_list', 'new_result_files_list'];
    listIds.forEach(id => {
        const fileList = document.getElementById(id);
        if (!fileList) return;
        fileList.querySelectorAll('button').forEach(btn => {
            const label = (btn.getAttribute('aria-label') || btn.getAttribute('title') || '').toLowerCase();
            const text = (btn.innerText || btn.textContent || '').trim().toLowerCase();
            if (label.includes('remove') || label.includes('clear') || label.includes('delete') || text === '✕' || text === 'x' || btn.classList.contains('label-clear-button')) {
                const row = btn.closest('tr') || btn.closest('.file-preview-item');
                if (row) {
                    const rowText = (row.innerText || row.textContent).toLowerCase();
                    const isImage = rowText.includes('.png') || rowText.includes('.jpg') || rowText.includes('.jpeg') || rowText.includes('.webp');
                    if (id === 'new_result_files_list' || isImage) {
                        btn.style.visibility = 'hidden';
                        btn.style.opacity = '0';
                        btn.style.pointerEvents = 'none';
                    } else {
                        btn.style.visibility = 'visible';
                        btn.style.opacity = '1';
                        btn.style.pointerEvents = '';
                    }
                }
            }
        });
    });
}, 500);

// 3. Status Polling Loop
(function () {
    let processingActive = false;
    let timerStartTime = null;
    let timerElement = null;
    let timerInterval = null;

    function createTimerElement() {
        if (!timerElement) {
            timerElement = document.createElement('div');
            timerElement.id = 'execution-timer';
            document.body.appendChild(timerElement);
        }
        return timerElement;
    }

    function startTimer() {
        timerStartTime = Date.now();
        const timer = createTimerElement();
        timer.classList.add('active');
        if (timerInterval) clearInterval(timerInterval);
        timerInterval = setInterval(() => {
            const elapsed = (Date.now() - timerStartTime) / 1000;
            const mins = Math.floor(elapsed / 60);
            const secs = Math.floor(elapsed % 60);
            timer.textContent = `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        }, 100);
    }

    function stopTimer() {
        if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
        if (timerElement) timerElement.classList.remove('active');
    }

    function setComponentState(id, waiting) {
        const el = document.getElementById(id);
        if (!el) return;

        if (waiting) {
            el.classList.add('waiting-component');
            // Force overflow on parent containers for textbox glow
            let parent = el.parentElement;
            while (parent && parent !== document.body) {
                parent.style.overflow = 'visible';
                parent = parent.parentElement;
            }

        } else {
            el.classList.remove('waiting-component');
            // Reset overflow
            let parent = el.parentElement;
            while (parent && parent !== document.body) {
                parent.style.overflow = '';
                parent = parent.parentElement;
            }
        }
    }

    function getLogText(id) {
        const el = document.getElementById(id);
        if (!el) return '';
        const tx = el.querySelector('textarea');
        return tx ? tx.value : '';
    }

    function setButtonAnimation(id, active) {
        const el = document.getElementById(id);
        if (el) {
            const btn = el.querySelector('button') || el;
            active ? btn.classList.add('btn-processing') : btn.classList.remove('btn-processing');
        }
    }

    function checkProcessingState() {
        const newLog = getLogText('new_job_log');
        const detLog = getLogText('det_job_log');

        const isNewJobProc = newLog.includes('[Training]') || newLog.includes('[50%]') || newLog.includes('[70%]');
        const isDetPlyProc = detLog.includes('[Generating]');
        const isDetVidProc = detLog.includes('[Rendering]');

        const isNewJobDone = newLog.includes('Completed:') || newLog.includes('[Done]') || newLog.includes('Error:') || newLog.includes('[Cancelled]');
        const isDetJobDone = detLog.includes('[Done]') || detLog.includes('Completed:') || detLog.includes('[Cancelled]');

        const activeProcess = (isNewJobProc && !isNewJobDone) || (isDetVidProc && !isDetJobDone) || (isDetPlyProc && !isDetJobDone);

        // Manage animations for Logs
        if (isNewJobProc && !isNewJobDone) {
            setComponentState('new_job_log', true);
        } else {
            setComponentState('new_job_log', false);
        }
        if ((isDetPlyProc || isDetVidProc) && !isDetJobDone) {
            setComponentState('det_job_log', true);
        } else {
            setComponentState('det_job_log', false);
        }

        if (activeProcess) {
            if (!processingActive) {
                processingActive = true;
                document.body.classList.add('ui-locked');
                startTimer();
                console.log("[UI] System Busy: Logic engaged.");

                if (isNewJobProc) setButtonAnimation('btn_start_gen', true);
                if (isDetPlyProc) setButtonAnimation('btn_gen_ply_details', true);
                if (isDetVidProc) setButtonAnimation('btn_gen_video_details', true);
            }
        } else if (processingActive && (isNewJobDone || isDetJobDone)) {
            processingActive = false;
            document.body.classList.remove('ui-locked');
            stopTimer();
            console.log("[UI] System Ready: Logic released.");

            setButtonAnimation('btn_start_gen', false);
            setButtonAnimation('btn_gen_ply_details', false);
            setButtonAnimation('btn_gen_video_details', false);
        }

        // Toggle STOP buttons based on processing state
        const stopNew = document.getElementById('btn_stop_gen_new');
        const stopDet = document.getElementById('btn_stop_gen_det');
        
        if (stopNew) {
            const btn = stopNew.querySelector('button') || stopNew;
            btn.disabled = !(isNewJobProc && !isNewJobDone);
        }
        if (stopDet) {
            const btn = stopDet.querySelector('button') || stopDet;
            btn.disabled = !((isDetPlyProc || isDetVidProc) && !isDetJobDone);
        }

        // --- NEW JOB TAB ---
        // 3D Model Pulse logic: Stop when PLY is ready OR when rendering starts
        const newPlyReady = newLog.includes('[50%]') || newLog.includes('PLY Generated') || newLog.includes('[70%]') || newLog.includes('[Rendering]');
        if (isNewJobProc && !newPlyReady && !isNewJobDone) {
            setComponentState('new_model_3d', true);
        } else {
            setComponentState('new_model_3d', false);
        }
        // Video Pulse logic: Keep until Job Done
        if (isNewJobProc && !isNewJobDone && newLog.includes('(Video: Enabled)')) {
            setComponentState('new_vid_color', true);
            setComponentState('new_vid_depth', true);
        } else {
            setComponentState('new_vid_color', false);
            setComponentState('new_vid_depth', false);
        }

        // --- HISTORY TAB ---
        // 3D Model Pulse logic: Stop when PLY is ready
        // DetPly means we are generating PLY. If we switch to Rendering, we are done with PLY.
        const detPlyReady = detLog.includes('PLY Generated') || isDetVidProc || detLog.includes('[Rendering]');
        if (isDetPlyProc && !detPlyReady && !isDetJobDone) {
            setComponentState('det_model_3d', true);
        } else {
            setComponentState('det_model_3d', false);
        }

        // Video Pulse logic
        if (isDetVidProc && !isDetJobDone) {
            setComponentState('det_vid_color', true);
            setComponentState('det_vid_depth', true);
        } else {
            setComponentState('det_vid_color', false);
            setComponentState('det_vid_depth', false);
        }

        // Final fallback for animation stop
        if (isNewJobDone) {
            setComponentState('new_model_3d', false);
            setComponentState('new_vid_color', false);
            setComponentState('new_vid_depth', false);
        }
        if (isDetJobDone) {
            setComponentState('det_model_3d', false);
            setComponentState('det_vid_color', false);
            setComponentState('det_vid_depth', false);
        }
    }
    setInterval(checkProcessingState, 500);
})();

function resetHistoryUI() {
    document.querySelectorAll('.job-list-item').forEach(el => el.classList.remove('selected'));
    const container = document.querySelector('.job-list-container');
    if (container) container.scrollTop = 0;
}