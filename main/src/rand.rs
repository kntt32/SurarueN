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
        seed |= 0xff00_ff00_ff00_ff00_00ff_00ff_00ff_00ff;
        seed ^= (seed_copy&0x01ff_ffff_ffff_ffff_ffff_ffff_ffff_ffff) << 7;
        seed ^= seed_copy >> 9;

        SEED = Some(seed);

        (seed & 0xffff_ffff_ffff_ffff) as u64
    }
}

