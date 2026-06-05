use alloc::{boxed::Box, vec::Vec};

use crate::lock::{rank, Mutex};

/// A pool of free [`wgpu_hal::CommandEncoder`]s, owned by a `Device`.
///
/// Each encoder in this list is in the "closed" state.
///
/// Since a raw [`CommandEncoder`][ce] is itself a pool for allocating
/// raw [`CommandBuffer`][cb]s, this is a pool of pools.
///
/// [`wgpu_hal::CommandEncoder`]: hal::CommandEncoder
/// [ce]: hal::CommandEncoder
/// [cb]: hal::Api::CommandBuffer
pub(crate) struct CommandAllocator {
    free_encoders: Mutex<Vec<Box<dyn hal::DynCommandEncoder>>>,
    /// Separate free-list for encoders created on the async-compute queue
    /// family (mesher∥render arc). A command buffer is bound to its pool's queue
    /// family, so a compute-family encoder must NEVER be recycled into the
    /// graphics `free_encoders` (it would be handed to a graphics submit).
    free_compute_encoders: Mutex<Vec<Box<dyn hal::DynCommandEncoder>>>,
}

impl CommandAllocator {
    pub(crate) fn new() -> Self {
        Self {
            free_encoders: Mutex::new(rank::COMMAND_ALLOCATOR_FREE_ENCODERS, Vec::new()),
            free_compute_encoders: Mutex::new(
                rank::COMMAND_ALLOCATOR_FREE_ENCODERS,
                Vec::new(),
            ),
        }
    }

    /// Return a fresh [`wgpu_hal::CommandEncoder`] in the "closed" state.
    ///
    /// If we have free encoders in the pool, take one of those. Otherwise,
    /// create a new one on `device`.
    ///
    /// [`wgpu_hal::CommandEncoder`]: hal::CommandEncoder
    pub(crate) fn acquire_encoder(
        &self,
        device: &dyn hal::DynDevice,
        queue: &dyn hal::DynQueue,
    ) -> Result<Box<dyn hal::DynCommandEncoder>, hal::DeviceError> {
        let mut free_encoders = self.free_encoders.lock();
        match free_encoders.pop() {
            Some(encoder) => Ok(encoder),
            None => unsafe {
                let hal_desc = hal::CommandEncoderDescriptor { label: None, queue };
                device.create_command_encoder(&hal_desc)
            },
        }
    }

    /// Add `encoder` back to the free pool.
    pub(crate) fn release_encoder(&self, encoder: Box<dyn hal::DynCommandEncoder>) {
        let mut free_encoders = self.free_encoders.lock();
        free_encoders.push(encoder);
    }

    /// Like [`Self::acquire_encoder`], but the encoder is created on the
    /// async-compute queue family (mesher∥render arc) so its command buffers can
    /// be submitted via `Queue::submit_compute`.
    pub(crate) fn acquire_encoder_compute(
        &self,
        device: &dyn hal::DynDevice,
        queue: &dyn hal::DynQueue,
    ) -> Result<Box<dyn hal::DynCommandEncoder>, hal::DeviceError> {
        let mut free = self.free_compute_encoders.lock();
        match free.pop() {
            Some(encoder) => Ok(encoder),
            None => unsafe {
                let hal_desc = hal::CommandEncoderDescriptor { label: None, queue };
                device.create_command_encoder_compute(&hal_desc)
            },
        }
    }

    /// Add a compute-family `encoder` back to the compute free pool.
    pub(crate) fn release_compute_encoder(&self, encoder: Box<dyn hal::DynCommandEncoder>) {
        let mut free = self.free_compute_encoders.lock();
        free.push(encoder);
    }
}
