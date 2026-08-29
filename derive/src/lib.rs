//! `#[derive(Coord)]` and `#[derive(Packed)]`.
//!
//! `std` gives a key type its address with `#[derive(Hash)]`. Without an equivalent, every user
//! type that wants to be an `apeiron` key needs a hand-written impl, and a survey of one real
//! codebase says the shapes are dull and repetitive: four newtypes over `u32`, one over `String`,
//! and nothing with more than one field. That is exactly the work a macro should be doing.
//!
//! # Why the generated code is not a forward to the tuple impls
//!
//! The obvious body for a three-field struct is `(self.a, self.b, self.c).coord()`, and it is what
//! a hand-written impl looks like. It does not generalise: `apeiron` implements `Coord` for tuples
//! of arity **2, 3 and 4 only**, so a five-field struct has nothing to forward to and a one-field
//! newtype has nothing either -- there is no `Coord` for a 1-tuple. So the derive emits the fold
//! directly, and matches the tuple impls step for step where they overlap, which
//! `tests/a_derived_key_is_a_key.rs` pins by comparing the two.
//!
//! # The contract, which is `Hash`'s contract
//!
//! **`a == b` must imply `a.coord() == b.coord()`.** A derive over all fields of a type whose `Eq`
//! is also derived satisfies it for free. `#[coord(skip)]` is where that can be broken, and it is
//! the caller's job to keep the two in step -- the same deal `Hash` offers, with the same
//! consequence for breaking it: lookups miss keys that are present.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as Ts;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Fields, Index, parse_macro_input};

/// Derive [`Coord`](../apeiron/coord/trait.Coord.html): the key's 64-bit address.
///
/// Folds every field, in declaration order, with the same `mix`/`finish` chain the tuple impls
/// use. Works on structs (named, tuple and unit), on enums with any mix of variant shapes, and on
/// generic types -- each field type picks up a `Coord` bound.
///
/// `#[coord(skip)]` leaves a field out. Use it only for a field that `Eq` also ignores.
///
/// [`Coord::SPREAD`](../apeiron/coord/trait.Coord.html#associatedconstant.SPREAD) is derived too,
/// as the conjunction over the participating fields: a newtype over `String` inherits it and skips
/// the redundant mix, a newtype over `u32` does not and keeps the spread that carries its pepper.
#[proc_macro_derive(Coord, attributes(coord))]
pub fn derive_coord(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match coord_body(&input) {
        Ok(body) => body.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// Derive [`Packed`](../apeiron/coord/trait.Packed.html): the *bijective* address, for a key that
/// fits in a word.
///
/// Only for a single-field newtype whose field is itself `Packed`. That restriction is the point:
/// `Packed` promises `unpack(k.packed()) == k` exactly, and the only shape where that can be
/// checked by construction rather than by argument is the one that just forwards. Anything else is
/// a compile error naming why.
///
/// It is worth having because a newtype over `u32` is what half the handle types in a real
/// codebase are, and a `Packed` key runs on the elliptic face -- `Words<BlockId, V>` stores no key
/// at all and never compares one.
#[proc_macro_derive(Packed)]
pub fn derive_packed(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match packed_body(&input) {
        Ok(body) => body.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// Whether a field carries `#[coord(skip)]`.
fn is_skipped(attrs: &[syn::Attribute]) -> syn::Result<bool> {
    let mut skip = false;
    for attr in attrs {
        if !attr.path().is_ident("coord") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("skip") {
                skip = true;
                Ok(())
            } else {
                Err(meta.error("unknown `coord` option; the only one is `skip`"))
            }
        })?;
    }
    Ok(skip)
}

/// The fold over a list of already-bound field expressions.
///
/// - **no fields**: a constant. Two unit structs are different types, so they never meet in one
///   table and the value only has to be stable, not distinctive.
/// - **one field**: the field's own address, untouched. A newtype *is* its field, and the table
///   avalanches the result anyway (`Salt::spread`), so wrapping it in a `finish` would be a
///   multiply and two shifts spent to no end.
/// - **more**: `finish(mix(mix(f0, f1), f2) ...)`, left to right, which is exactly what
///   `impl Coord for (A, B, C)` does.
fn fold_over(exprs: &[Ts], pepper: &Ts) -> Ts {
    match exprs {
        [] => quote! { ::apeiron::coord::PHI },
        [one] => quote! { ::apeiron::coord::Coord::coord_with(#one, #pepper) },
        [first, rest @ ..] => {
            let mut acc = quote! { ::apeiron::coord::Coord::coord_with(#first, #pepper) };
            for e in rest {
                acc = quote! {
                    ::apeiron::coord::mix(
                        #acc,
                        ::apeiron::coord::Coord::coord_with(#e, #pepper),
                    )
                };
            }
            quote! { ::apeiron::coord::finish(#acc) }
        }
    }
}

/// The field expressions of a struct's fields, as `&self.x` / `&self.0`, minus the skipped ones.
fn struct_fields(fields: &Fields) -> syn::Result<(Vec<Ts>, Vec<syn::Type>)> {
    let mut exprs = Vec::new();
    let mut types = Vec::new();
    match fields {
        Fields::Named(named) => {
            for f in &named.named {
                if is_skipped(&f.attrs)? {
                    continue;
                }
                let name = f.ident.as_ref().expect("named field has an ident");
                exprs.push(quote! { &self.#name });
                types.push(f.ty.clone());
            }
        }
        Fields::Unnamed(unnamed) => {
            for (i, f) in unnamed.unnamed.iter().enumerate() {
                if is_skipped(&f.attrs)? {
                    continue;
                }
                let idx = Index::from(i);
                exprs.push(quote! { &self.#idx });
                types.push(f.ty.clone());
            }
        }
        Fields::Unit => {}
    }
    Ok((exprs, types))
}

fn coord_body(input: &DeriveInput) -> syn::Result<Ts> {
    let name = &input.ident;
    let pepper = quote! { __apeiron_pepper };

    let (body, field_types) = match &input.data {
        Data::Struct(s) => {
            let (exprs, types) = struct_fields(&s.fields)?;
            (fold_over(&exprs, &pepper), types)
        }
        Data::Union(u) => {
            return Err(syn::Error::new_spanned(
                u.union_token,
                "Coord cannot be derived for a union: which field is live is not knowable here, \
                 and an address that reads the wrong one is not a function of the key",
            ));
        }
        Data::Enum(e) => {
            let mut types = Vec::new();
            let mut arms = Vec::new();
            for (i, variant) in e.variants.iter().enumerate() {
                let vname = &variant.ident;
                // The variant index seeds the fold, so two variants carrying equal payloads --
                // `A(7)` and `B(7)` -- do not share an address.
                let seed = quote! { ::apeiron::coord::mix(#i as u64, ::apeiron::coord::PHI) };
                let (pat, exprs) = match &variant.fields {
                    Fields::Unit => (quote! { Self::#vname }, Vec::new()),
                    Fields::Unnamed(unnamed) => {
                        let mut binds = Vec::new();
                        let mut exprs = Vec::new();
                        for (j, f) in unnamed.unnamed.iter().enumerate() {
                            let b = format_ident!("__f{}", j);
                            binds.push(quote! { #b });
                            if !is_skipped(&f.attrs)? {
                                exprs.push(quote! { #b });
                                types.push(f.ty.clone());
                            }
                        }
                        (quote! { Self::#vname( #(#binds),* ) }, exprs)
                    }
                    Fields::Named(named) => {
                        let mut binds = Vec::new();
                        let mut exprs = Vec::new();
                        for f in &named.named {
                            let id = f.ident.as_ref().expect("named field has an ident");
                            binds.push(quote! { #id });
                            if !is_skipped(&f.attrs)? {
                                exprs.push(quote! { #id });
                                types.push(f.ty.clone());
                            }
                        }
                        (quote! { Self::#vname { #(#binds),* } }, exprs)
                    }
                };
                // Every field folds onto the variant seed, so the discriminant is never dropped --
                // even for a variant whose fields were all skipped.
                let mut acc = seed;
                for e in &exprs {
                    acc = quote! {
                        ::apeiron::coord::mix(
                            #acc,
                            ::apeiron::coord::Coord::coord_with(#e, #pepper),
                        )
                    };
                }
                arms.push(quote! { #pat => ::apeiron::coord::finish(#acc) });
            }
            if arms.is_empty() {
                // An uninhabited enum has no values, so `coord` can never be called -- but the impl
                // still has to typecheck.
                (
                    quote! { match *self {} },
                    Vec::new(),
                )
            } else {
                (quote! { match self { #(#arms),* } }, types)
            }
        }
    };

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    // Bound the *field* types rather than the type parameters. `struct S<T> { v: Vec<T> }` needs
    // `Vec<T>: Coord`, which a blanket `T: Coord` neither implies nor is implied by.
    // Deduplicated by rendered tokens: `syn::Type` has no `PartialEq` without `extra-traits`, and
    // a duplicate bound is a hard error rather than a wasted line.
    let mut bounds: Vec<syn::Type> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    for t in field_types {
        let rendered = quote! { #t }.to_string();
        if !seen.contains(&rendered) {
            seen.push(rendered);
            bounds.push(t);
        }
    }
    let where_clause = if bounds.is_empty() {
        quote! { #where_clause }
    } else {
        match where_clause {
            Some(w) => quote! { #w #(#bounds: ::apeiron::coord::Coord,)* },
            None => quote! { where #(#bounds: ::apeiron::coord::Coord,)* },
        }
    };

    // `Coord::SPREAD` says the fold's output is already mixed, so the parabolic face may skip
    // `Salt::spread`. Left at its `false` default -- which is what this derive used to do -- a
    // newtype over `String` paid a redundant widening multiply on every lookup that a hand-written
    // `impl Coord for String` does not, which is most of a 0.87 ns per-lookup constant.
    //
    // The rule is conjunction over every field of every variant, and it is stricter than soundness
    // needs. Skipping the spread moves where the **pepper** enters -- from the bijection's
    // multiplier into the fold's own seed -- so the obligation is that `coord_with` consumes the
    // pepper, and *one* field doing so already suffices because `mix`/`finish` carry it into the
    // whole. Conjunction was chosen anyway: it is obviously right without a case analysis, it
    // agrees with disjunction on the shape that matters (a single-field newtype), and a mixed
    // composite that wants the sharper answer can write the const by hand. A wrong `true` here does
    // not fail a test -- it compiles, spreads fine, and silently drops the pepper.
    //
    // Zero fields is `false`: a unit struct folds to a constant, which no pepper reaches.
    let spread = if bounds.is_empty() {
        quote! { false }
    } else {
        let mut acc = quote! { true };
        for t in &bounds {
            acc = quote! { #acc && <#t as ::apeiron::coord::Coord>::SPREAD };
        }
        acc
    };

    Ok(quote! {
        #[automatically_derived]
        impl #impl_generics ::apeiron::coord::Coord for #name #ty_generics #where_clause {
            const SPREAD: bool = #spread;
            #[inline]
            fn coord(&self) -> u64 {
                <Self as ::apeiron::coord::Coord>::coord_with(self, 0)
            }
            #[inline]
            fn coord_with(&self, #pepper: u64) -> u64 {
                #body
            }
        }
    })
}

fn packed_body(input: &DeriveInput) -> syn::Result<Ts> {
    let name = &input.ident;
    let refuse = |span: proc_macro2::Span| {
        syn::Error::new(
            span,
            "Packed can only be derived for a newtype -- one field, itself Packed. It promises \
             `unpack(k.packed()) == k` exactly, and forwarding to a single field is the only shape \
             where that holds by construction rather than by argument. Write the impl by hand, or \
             derive Coord instead and use the parabolic face.",
        )
    };

    let Data::Struct(s) = &input.data else {
        return Err(refuse(name.span()));
    };
    let (build, field_ty) = match &s.fields {
        Fields::Unnamed(u) if u.unnamed.len() == 1 => {
            let ty = u.unnamed[0].ty.clone();
            (quote! { Self(__v) }, ty)
        }
        Fields::Named(n) if n.named.len() == 1 => {
            let f = &n.named[0];
            let id = f.ident.as_ref().expect("named field has an ident");
            (quote! { Self { #id: __v } }, f.ty.clone())
        }
        _ => return Err(refuse(name.span())),
    };
    let read = match &s.fields {
        Fields::Unnamed(_) => quote! { &self.0 },
        _ => {
            let id = match &s.fields {
                Fields::Named(n) => n.named[0].ident.clone().expect("named"),
                _ => unreachable!("checked above"),
            };
            quote! { &self.#id }
        }
    };

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let where_clause = match where_clause {
        Some(w) => quote! { #w #field_ty: ::apeiron::coord::Packed, },
        None => quote! { where #field_ty: ::apeiron::coord::Packed, },
    };

    Ok(quote! {
        #[automatically_derived]
        impl #impl_generics ::apeiron::coord::Packed for #name #ty_generics #where_clause {
            #[inline]
            fn packed(&self) -> u64 {
                ::apeiron::coord::Packed::packed(#read)
            }
            #[inline]
            fn unpack(__c: u64) -> Self {
                let __v = <#field_ty as ::apeiron::coord::Packed>::unpack(__c);
                #build
            }
        }
    })
}
