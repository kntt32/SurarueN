use std::time::SystemTime;

pub fn rand() -> u64 {
    static mut SEED: Option<u128> = None;

    unsafe {
        let mut seed = if let None = SEED {
            SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_millis()
        }else {
            SEED.unwrap()
        };

        let seed_copy = seed;
        
        seed ^= (seed_copy&0x01ff_ffff_ffff_ffff_ffff_ffff_ffff_ffff) << 7;
        seed ^= seed_copy >> 9;

        SEED = Some(seed);

        (seed & 0xffff_ffff_ffff_ffff) as u64
    }
}


pub fn shuffle<T>(target: &mut [T]) -> &mut [T] {
    for i in (1 .. target.len()).rev() {
        let swap_index = (((rand() & 0xffff_ffff) as usize) % i);
        target.swap(swap_index, i);
    }

    target
}

