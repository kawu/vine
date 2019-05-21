{-# LANGUAGE OverloadedStrings #-}


module DhallUtils
  ( rmUnion_1

  -- * Double operations
  -- , doubleSettings
  ) where


import           Control.Monad (guard)

import           Dhall -- (Interpret)
import qualified Dhall.Map as Map
import           Dhall.Core (Expr(..))

-- import Control.Applicative
-- import Data.String (IsString, fromString)
-- 
-- import Dhall.Core (Expr(..), ReifiedNormalizer(..))
-- import qualified Dhall.Core
-- 
-- import qualified Data.Text (Text(..))
-- import qualified Data.Text.IO
-- import qualified Dhall as Dhall
-- import qualified Dhall.Context
-- -- import qualified Lens.Family   as Lens
-- import Data.Functor.Identity (Identity(..))

-- import Debug.Trace (trace)


----------------------------------------------
-- Removing the top-level _1 fields
----------------------------------------------


-- | Remove the top-level _1 fields from the given union type.
rmUnion_1 :: Type a -> Type a
rmUnion_1 typ = typ
  { extract = \expr -> extract typ (add1 expr)
  , expected = rm1 (expected typ)
  }
  where
    -- Add _1 to union expression
    add1 expr =
      case expr of
        Union m -> Union (fmap addField_1 m)
        UnionLit key val m -> UnionLit key (addField_1 val) (fmap addField_1 m)
        _ -> expr
    -- Remove _1 from union epxression
    rm1 expr =
      case expr of
        Union m -> Union (fmap rmField_1 m)
        UnionLit key val m -> UnionLit key (rmField_1 val) (fmap rmField_1 m)
        _ -> expr


-- | Add _1 in the given record expression.
addField_1 :: Expr s a -> Expr s a
addField_1 expr =
  case expr of
    RecordLit m -> RecordLit (Map.singleton "_1" (RecordLit m))
    Record m -> Record (Map.singleton "_1" (Record m))
    _ -> expr


-- | Remove _1 from the given record expression.
rmField_1 :: Expr s a -> Expr s a
rmField_1 expr =
  case expr of
    RecordLit m -> Prelude.maybe (RecordLit m) id $ do
      guard $ Map.keys m == ["_1"]
      RecordLit m' <- Map.lookup "_1" m
      return (RecordLit m')
    Record m -> Prelude.maybe (Record m) id $ do
      guard $ Map.keys m == ["_1"]
      Record m' <- Map.lookup "_1" m
      return (Record m')
    _ -> expr


-- ----------------------------------------------------------------------
-- NOTE: The code below, commented out, is the trace of the effort put in
-- making it possible to specify the "gradually decreasing stepsize" as a
-- function directly in the .dhall configuration file.
--
--
-- ----------------------------------------------
-- -- Double settings
-- ----------------------------------------------
-- 
-- 
-- doubleSettings :: (InputSettings, InterpretOptions)
-- doubleSettings =
--   let context     = doubleEnrichedContext
--       normalizer  = ReifiedNormalizer $ pure . doubleNormalizer
--       addSettings = Lens.set Dhall.normalizer normalizer
--                   . Lens.set Dhall.startingContext context
--       inputSettings = addSettings Dhall.defaultInputSettings 
--       interpretOptions =
--           Dhall.defaultInterpretOptions
--             { Dhall.inputNormalizer = normalizer }
--    in (inputSettings, interpretOptions)
-- 
-- 
-- ----------------------------------------------
-- -- Double operations
-- ----------------------------------------------
-- 
-- 
-- binaryDoubleType :: Expr s a
-- binaryDoubleType = Pi "_" Double (Pi "_" Double Double)
-- 
-- -- | Make a normalizer function from a Haskell function and its Dhall name,
-- -- e.g.
-- --
-- -- >>> :t binaryDoubleNormalizer (+) "Double/add"
-- -- binaryDoubleNormalizer (+) "Double/add" :: Expr s1 a1 -> Maybe (Expr s2 a2)
-- binaryDoubleNormalizer
--   :: (Double -> Double -> Double)
--   -> Dhall.Core.Var
--   -> Expr s1 a1
--   -> Maybe (Expr s2 a2)
-- binaryDoubleNormalizer f name (App (App (Var match) (DoubleLit x)) (DoubleLit y))
--   | name == match = trace (show (x, y)) $ Just (DoubleLit (f x y))
--   | otherwise = Nothing
-- binaryDoubleNormalizer _ _ _ = Nothing
-- 
-- doubleFunctions :: IsString s => [(s, Double -> Double -> Double)]
-- doubleFunctions =
--   [ ("Double/add", (+))
--   , ("Double/sub", (-))
--   , ("Double/mul", (*))
--   , ("Double/div", (/))
--   ]
-- 
-- -- | A 'Dhall.Context.Context' with double
-- {-doubleEnrichedContext :: [Dhall.Context.Context a]-}
-- doubleEnrichedContext = foldl f Dhall.Context.empty doubleFunctions
--   where f ctx (name, _) = Dhall.Context.insert name binaryDoubleType ctx
-- 
-- -- utility to try each function in a list of functions until one of them
-- -- succeeeds.
-- tryAll :: (Functor t, Foldable t) => t (a -> Maybe b) -> a -> Maybe b
-- tryAll fs x = foldl (<|>) Nothing $ fmap ($x) fs
-- 
-- doubleNormalizer :: Expr s1 a1 -> Maybe (Expr s2 a2)
-- doubleNormalizer =
--   tryAll [ binaryDoubleNormalizer f n | (n, f) <- doubleFunctions ]
