use std::sync::Arc;
use tokio::sync::Mutex;

/// History structure to store chat history
pub struct History {
    records: Arc<Mutex<Vec<String>>>,
    current_index: Arc<Mutex<usize>>,
}

impl History {
    /// Creates a new History instance
    pub fn new() -> Self {
        Self {
            records: Arc::new(Mutex::new(Vec::new())),
            current_index: Arc::new(Mutex::new(0)),
        }
    }

    /// Adds a new record to the history
    pub async fn add(&self, record: String) {
        let mut records = self.records.lock().await;
        records.push(record);
        let mut current_index = self.current_index.lock().await;
        *current_index = records.len();
    }

    /// Gets the previous record from history
    pub async fn get_previous(&self) -> Option<String> {
        let records = self.records.lock().await;
        let mut current_index = self.current_index.lock().await;
        if *current_index > 0 {
            *current_index -= 1;
            records.get(*current_index).cloned()
        } else {
            None
        }
    }

    /// Gets the next record from history
    pub async fn get_next(&self) -> Option<String> {
        let records = self.records.lock().await;
        let mut current_index = self.current_index.lock().await;
        if *current_index < records.len() - 1 {
            *current_index += 1;
            records.get(*current_index).cloned()
        } else {
            None
        }
    }

    /// Gets the number of records in history
    pub async fn len(&self) -> usize {
        self.records.lock().await.len()
    }
}
