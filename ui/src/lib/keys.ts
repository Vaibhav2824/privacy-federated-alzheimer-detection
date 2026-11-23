/**
 * Pairwise mask seeds for secure aggregation, agreed with ECDH (P-256) in the
 * browser. Each hospital publishes only its public key; two hospitals derive
 * the same shared secret, and hash it with the round number into a 32-bit
 * mask seed. The server relays public keys and never learns any seed.
 */

const CURVE = { name: 'ECDH', namedCurve: 'P-256' } as const;

function toBase64(bytes: ArrayBuffer): string {
  return btoa(String.fromCharCode(...new Uint8Array(bytes)));
}

function fromBase64(text: string): Uint8Array<ArrayBuffer> {
  return Uint8Array.from(atob(text), (c) => c.charCodeAt(0));
}

export interface KeyPair { privateKey: CryptoKey; publicKey: string }

export async function newKeyPair(): Promise<KeyPair> {
  const pair = await crypto.subtle.generateKey(CURVE, false, ['deriveBits']);
  return { privateKey: pair.privateKey, publicKey: toBase64(await crypto.subtle.exportKey('raw', pair.publicKey)) };
}

/** The secret shared with one peer, from our private key and its public key. */
export async function sharedSecret(privateKey: CryptoKey, peerPublic: string): Promise<Uint8Array> {
  const peer = await crypto.subtle.importKey('raw', fromBase64(peerPublic), CURVE, false, []);
  return new Uint8Array(await crypto.subtle.deriveBits({ name: 'ECDH', public: peer }, privateKey, 256));
}

/** A fresh 32-bit mask seed for one round, the same on both sides of the pair. */
export async function roundSeed(secret: Uint8Array, round: number): Promise<number> {
  const input = new Uint8Array(secret.length + 4);
  input.set(secret);
  new DataView(input.buffer).setUint32(secret.length, round);
  const digest = new DataView(await crypto.subtle.digest('SHA-256', input));
  return digest.getUint32(0);
}
