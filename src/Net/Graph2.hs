{-# LANGUAGE CPP #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections #-}

{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}

{-# LANGUAGE PolyKinds #-}

{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE Rank2Types #-}

{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeFamilies #-}

-- To automatically derive Binary
{-# LANGUAGE DeriveAnyClass #-}


-- | Processing graphs with artificial neural networks.  This module allows to
-- assign labels to both arcs (as in `Net.Graph`) and nodes (in contrast to
-- `Net.Graph`).
--
-- The plan is to play with the arc-factored model first.  Two inference
-- methods can be considered: let's call them local and global.  In either
-- case, we assign potentials two combinations of values:
--
--   * (label of the head, label of the arc, label of the dependent)
--
-- In local inference, we split this assignment to make the potentials of:
--
--   * label of the head
--   * label of the arc
--   * label of the dependent
--
-- completely separate.  Then we can simply sum the potentials before making
-- the final decision as to whether a given node should be considered an MWE
-- component or not.  Again, this makes the decisions for the individual nodes
-- and arcs independent (at least intuitively, I'm not sure if formally these
-- decisions are completely independent).
--
-- In case of global inference, we want to maximize the total potential of a
-- labeling of the whole dependency tree.  We can certainly do that at the time
-- of tagging, not sure how/if this could work during training.


module Net.Graph2
  ( 
  -- * Network
    new
  , run
  , eval

  -- * Opaque
  , Opaque(..)
  , Typ(..)
  , newO

  -- * Serialization
  , save
  , load

  -- * Data set
  , DataSet
  , Elem(..)

  -- * Error
  , netError

  -- * Encoding
  , Out(..)
  , encode
  , decode
  , rightInTwo

  -- * Explicating
  , enumerate
  , explicate
  , obfuscate

  -- * Inference
  , tagGreedy
  ) where


import           GHC.Generics (Generic)
import           GHC.TypeNats (KnownNat, natVal)
-- import           GHC.Natural (naturalToInt)
import qualified GHC.TypeNats as Nats
import           GHC.TypeLits (Symbol, KnownSymbol)

import           System.Random (randomRIO)

import           Control.Monad (forM_, forM)

import           Lens.Micro.TH (makeLenses)
-- import           Lens.Micro ((^.))

import           Data.Proxy (Proxy(..))
-- import           Data.Ord (comparing)
import qualified Data.List as List
import           Data.Maybe (catMaybes, mapMaybe)
import qualified Data.Text as T
import qualified Data.Set as S
import qualified Data.Map.Strict as M
import qualified Data.Graph as G
import qualified Data.Array as A
import           Data.Binary (Binary)
import qualified Data.Binary as Bin
import qualified Data.ByteString.Lazy as BL
import           Codec.Compression.Zlib (compress, decompress)

-- import qualified Text.Read as R
-- import qualified Text.ParserCombinators.ReadP as R

-- import qualified Data.Map.Lens as ML
import           Control.Lens.At (ixAt)
import           Control.Lens (Lens)
import           Control.DeepSeq (NFData)

import           Dhall (Interpret)
import qualified Data.Aeson as JSON

import qualified Prelude.Backprop as PB
import qualified Numeric.Backprop as BP
import           Numeric.Backprop (Backprop, (^^.), (^^?))
import qualified Numeric.LinearAlgebra.Static.Backprop as LBP
import           Numeric.LinearAlgebra.Static.Backprop
  (R, L, BVar, Reifies, W, (#), (#>), dot)
import qualified Numeric.LinearAlgebra.Static as LA

import qualified Test.SmallCheck.Series as SC

import           Net.New
import           Net.Pair
import           Net.Util
-- import qualified Net.List as NL
import qualified Net.FeedForward as FFN
import           Net.FeedForward (FFN(..))
-- import qualified GradientDescent as GD
-- import qualified GradientDescent.Momentum as Mom
-- import qualified GradientDescent.Nestorov as Mom
-- import qualified GradientDescent.AdaDelta as Ada
import           Numeric.SGD.ParamSet (ParamSet)

import           Graph
import           Net.Graph2.BiComp (Pot, Prob, Vec(..), Vec8)
import qualified Net.Graph2.BiComp as B
-- import qualified Net.Graph.QuadComp as Q

import           Debug.Trace (trace)


----------------------------------------------
-- Opaque
----------------------------------------------


-- | Each constructor of `Typ` corresponds to a specific network architecture.
data Typ
  = Arc0T
  | Arc1T
  deriving (Generic, Binary, Read)


-- | Opaque neural network with the actual architecture abstracted away.
-- The type (`Typ`) is kept for the sake of (de-)serialization.
--
-- `Opaque` is motivated by the fact that each network architecture is
-- represeted by a different type.  We know, however, that each architecture is
-- an instance of `B.QuadComp`, `ParamSet`, etc., and this is precisely
-- what `Opaque` preserves.
data Opaque :: Nats.Nat -> * -> * -> * where
  Opaque 
    :: (Binary p, NFData p, ParamSet p, B.BiComp d a b p)
    => Typ -> p -> Opaque d a b


instance ( KnownNat d
         , Binary a, Binary b, NFData a, NFData b
         , Ord a, Ord b, Show a, Show b
         )
  => Binary (Opaque d a b) where
  put (Opaque typ p) = Bin.put typ >> Bin.put p
  get = do
    typ <- Bin.get
    exec typ Bin.get


-- | Execute the action within the context of the given model type.  Just a
-- helper function, really, which allows to avoid boilerplate code.
exec
  :: forall d a b m.
    ( KnownNat d
    , Binary a, NFData a, Show a, Ord a
    , Binary b, NFData b, Show b, Ord b
    , Functor m
    )
  => Typ
  -> (forall p. (Binary p, New a b p) => m p)
  -> m (Opaque d a b)
exec typ action =
  case typ of
    Arc0T -> Opaque typ <$> (action :: m (Arc0 d a b))
    Arc1T -> Opaque typ <$> (action :: m (Arc1 d a b))


-- | Create a new network of the given type.
newO
  :: forall d a b.
    ( KnownNat d
    , Binary a, NFData a, Show a, Ord a
    , Binary b, NFData b, Show b, Ord b
    )
  => Typ     -- ^ Network type, which determines its architecture
  -> S.Set a -- ^ The set of node labels
  -> S.Set b -- ^ The set of arc labels
  -> IO (Opaque d a b)
newO typ xs ys =
  exec typ (new xs ys)


----------------------------------------------
-- Different network architectures
----------------------------------------------


-- | Arc-factored model (0)
type Arc0 d a b
   = B.BiAff d 100
  :& B.Bias


-- | Arc-factored model (1)
type Arc1 d a b
   = B.BiAff d 100
  :& BiParam a b


-- -- | Arc-factored model (2)
-- type Arc2 d a b
--    = Q.BiAffExt d 50 a b 100
--   :& Q.BiQuad (BiParam a b)
-- 
-- 
-- -- | Arc-factored model (3)
-- type Arc3 d a b
--   = Q.BiQuad (PotArc d 100 a b)
-- 
-- 
-- -- | Arc-factored model (4); similar to (3), but with (h = d).
-- -- The best for LVC.full in French found so far!
-- type Arc4 d a b
--   = Q.BiQuad (PotArc d d a b)
-- 
-- 
-- -- | Arc-factored model (5): (2) - `BiParam`
-- type Arc5 d a b
--    = Q.BiAffExt d 50 a b 100
-- 
-- 
-- -- | Arc-factored model (6): (2) + `Q.UnordBiAff`
-- type Arc6 d a b
--    = Q.BiAffExt d 50 a b 100
--   :& Q.UnordBiAff d 100
--   :& Q.BiQuad (BiParam a b)


-- | Basic bi-affine compoments (easy to compute, based on POS and DEP labels
-- exclusively)
type BiParam a b 
   = B.Bias
  :& B.HeadPosAff a
  :& B.DepPosAff a
  :& B.PapyPosAff a
  :& B.EnkelPosAff a
  :& B.ArcBias b
  :& B.PapyArcAff b
  :& B.EnkelArcAff b


-- -- | The best arc-factored model you found so far (with h = dim).
-- type PotArc dim h a b
--    = B.Biaff dim h
--   :& B.UnordBiaff dim h
--   :& B.HeadAff dim h
--   :& B.DepAff dim h
--   :& BiParam a b
-- 
-- 
-- -- | Quad-factored model (0) with hidden layers of size 100
-- type Quad0 d a b = 
--   QuadH d 100 a b
-- 
-- 
-- -- | Quad-factored model (1); (0) + unordered bi-affine component.  That's the
-- -- last model you tried on cl-srv2.  Turned out better than `Quad0`, the
-- -- `Q.UnordBiAff` seems to make a difference.
-- --
-- -- Now you could try to do some ablation/enriching study:
-- --
-- --   * What if you remove `Q.TriAff` and `Q.SibAff`?
-- --   * What if you also remove `Q.UnAff`?
-- --
-- -- You can also try `Arc3`, a simplified (with dim = 100) version of the best
-- -- model obtained so far (`Arc4`).  Just to see how much you potentialy (well,
-- -- it can depend on training anyway...) lose by using smaller dimension.
-- --
-- -- The next thing to do would be to check if you can gain somethin by using
-- -- POS/DEP embeddings.  For instance by enriching `Arc3` or `Arc4` with
-- -- `Q.BiAffExt` (see `Arc2`).  In fact, it makes sense to test `Arc1`, `Arc2`,
-- -- and `Arc3` first and see what they give.
-- --
-- type Quad1 d a b
--    = QuadH d 100 a b
--   :& Q.UnordBiAff d 100
-- 
-- 
-- -- | Quad-factored model with underspecified size of the hidden layers
-- type QuadH d h a b
--    = Q.TriAff d h
--   :& Q.SibAff d h
--   :& Q.BiAff d h
--   :& Q.UnAff d h
--   :& Q.Bias


----------------------------------------------
-- Evaluation
----------------------------------------------


-- | Run the given (bi-affine) network over the given graph within the context
-- of back-propagation.
runRaw
  :: ( KnownNat dim, Ord a, Show a, Ord b, Show b, Reifies s W
     , B.BiComp dim a b comp
     )
  => BVar s comp
    -- ^ Graph network / params
  -> Graph (Node dim a) b
    -- ^ Input graph labeled with initial hidden states
  -> M.Map Arc (BVar s (Vec8 Pot))
    -- ^ Output map with output potential values
runRaw net graph = M.fromList $ do
  arc <- graphArcs graph
  let x = B.runBiComp graph arc net
  return (arc, x)


-- | `runRaw` + softmax
run
  :: ( KnownNat dim, Ord a, Show a, Ord b, Show b, Reifies s W
     , B.BiComp dim a b comp
     )
  => BVar s comp
    -- ^ Graph network / params
  -> Graph (Node dim a) b
    -- ^ Input graph labeled with initial hidden states
  -> M.Map Arc (BVar s (Vec8 Prob))
    -- ^ Output map with output potential values
run net = fmap B.softmaxVec . runRaw net


-- | Evaluate the network over the given graph.  User-friendly (and without
-- back-propagation) version of `runRaw`.
evalRaw
  :: ( KnownNat dim, Ord a, Show a, Ord b, Show b
     , B.BiComp dim a b comp
     )
  => comp
    -- ^ Graph network / params
  -> Graph (Node dim a) b
    -- ^ Input graph labeled with initial hidden states (word embeddings)
  -> M.Map Arc (Vec8 Pot)
evalRaw net graph =
  BP.evalBP0
    ( BP.collectVar
    $ runRaw
        (BP.constVar net)
        graph
    )


-- | Evaluate the network over the given graph.  User-friendly (and without
-- back-propagation) version of `run`.
--
-- TODO: lot's of code common with `evalRaw`.
--
eval
  :: ( KnownNat dim, Ord a, Show a, Ord b, Show b
     , B.BiComp dim a b comp
     )
  => comp
    -- ^ Graph network / params
  -> Graph (Node dim a) b
    -- ^ Input graph labeled with initial hidden states (word embeddings)
  -> M.Map Arc (Vec8 Prob)
eval net graph =
  BP.evalBP0
    ( BP.collectVar
    $ run
        (BP.constVar net)
        graph
    )


----------------------------------------------
-- Inference
----------------------------------------------


-- -- | Graph labeling
-- data Labeling = Labeling
--   { nodLab :: M.Map G.Vertex Bool
--   , arcLab :: M.Map Arc Bool
--   }


-- | Greedily pick an arc labeling with high potential based on the given
-- potential map.
tagGreedy
  :: Double -- ^ Probability threshold (e.g. @0.5@)
  -> M.Map Arc (Vec8 Prob)
  -> M.Map Arc Bool
-- tagGreedy th m = M.fromList $ do
--   (e, vec) <- M.toList m
--   return (e, arcVal (decode vec) >= th)
tagGreedy th = fmap $ \vec -> arcVal (decode vec) >= th


----------------------------------------------
-- Explicating
----------------------------------------------


-- | Determine the values assigned to different labellings of the given arc and
-- nodes.
explicate :: Vec8 p -> M.Map (Out Bool) Double
explicate = M.fromList . zip enumerate . toList . unVec


-- | The inverse of `explicate`.
obfuscate :: M.Map (Out Bool) Double -> Vec8 p
obfuscate = Vec . LA.vector . M.elems


-- | Enumerate the possible arc/node labelings in order consistent with the
-- encoding/decoding format.
enumerate :: [Out Bool]
enumerate = do
  b1 <- [False, True]
  b2 <- [False, True]
  b3 <- [False, True]
  return $ Out b1 b2 b3


----------------------------------------------
-- Serialization
----------------------------------------------


-- | Save the parameters in the given file.
save :: (Binary a) => FilePath -> a -> IO ()
save path =
  BL.writeFile path . compress . Bin.encode


-- | Load the parameters from the given file.
load :: (Binary a) => FilePath -> IO a
load path =
  Bin.decode . decompress <$> BL.readFile path


----------------------------------------------
-- DataSet
----------------------------------------------


-- | Dataset element
data Elem dim a b = Elem
  { graph :: Graph (Node dim a) b
    -- ^ Input graph
  , arcMap :: M.Map Arc Double
    -- ^ Target arc probabilities
  , nodMap :: M.Map G.Vertex Double
    -- ^ Target node probabilities
  } deriving (Show, Generic, Binary)


-- | DataSet: a list of dataset elements
type DataSet d a b = [Elem d a b]


----------------------------------------------
-- Error
----------------------------------------------


-- | Cross entropy between the true and the artificial distributions
crossEntropy
  :: forall n s. (KnownNat n, Reifies s W)
  => BVar s (Vec n Prob)
    -- ^ Target ,,true'' distribution
  -> BVar s (Vec n Prob)
    -- ^ Output ,,artificial'' distribution
  -> BVar s Double
crossEntropy p0 q0 =
  negate (p `dot` LBP.vmap' log q)
  where
    p = BP.coerceVar p0 :: BVar s (R n)
    q = BP.coerceVar q0


-- | Cross entropy between the target and the output values
errorOne
  :: (Ord a, Reifies s W)
  => M.Map a (BVar s (Vec8 Prob))
    -- ^ Target values
  -> M.Map a (BVar s (Vec8 Prob))
    -- ^ Output values
  -> BVar s Double
errorOne target output = PB.sum . BP.collectVar $ do
  (key, tval) <- M.toList target
  let oval = output M.! key
  return $ crossEntropy tval oval


-- | Error on a dataset.
errorMany
  :: (Ord a, Reifies s W)
  => [M.Map a (BVar s (Vec8 Prob))] -- ^ Targets
  -> [M.Map a (BVar s (Vec8 Prob))] -- ^ Outputs
--   => [M.Map a (BVar s (R n))] -- ^ Targets
--   -> [M.Map a (BVar s (R n))] -- ^ Outputs
-- --   => [M.Map a (BVar s Double)] -- ^ Targets
-- --   -> [M.Map a (BVar s Double)] -- ^ Outputs
  -> BVar s Double
errorMany targets outputs =
  go targets outputs
  where
    go ts os =
      case (ts, os) of
        (t:tr, o:or) -> errorOne t o + go tr or
        ([], []) -> 0
        _ -> error "errorMany: lists of different size"


-- | Network error on a given dataset.
netError
  :: ( Reifies s W, KnownNat dim
     , Ord a, Show a, Ord b, Show b
     , B.BiComp dim a b comp
     )
  => DataSet dim a b
  -> BVar s comp
  -> BVar s Double
netError dataSet net =
  let
    inputs = map graph dataSet
    outputs = map (run net) inputs
    targets = map (fmap BP.auto . mkTarget) dataSet
  in
    errorMany targets outputs


-- | Create the target map from the given dataset element.
mkTarget
  :: Elem dim a b
  -> M.Map Arc (Vec8 Prob)
mkTarget el = M.fromList $ do
  ((v, w), arcP) <- M.toList (arcMap el)
  let vP = nodMap el M.! v
      wP = nodMap el M.! w
      target = Out
        { arcVal = arcP
        , hedVal = wP
        , depVal = vP }
  return ((v, w), encode target)


----------------------------------------------
-- Encoding
----------------------------------------------


-- | Output structure
data Out a = Out
  { arcVal :: a
    -- ^ Probability/potential of the arc
  , hedVal :: a
    -- ^ Probability/potential of the head
  , depVal :: a
    -- ^ Probability/potential of the dependent
  } deriving (Generic, Show, Eq, Ord)

-- Allows to use SmallCheck to test (decode . encode) == id.
instance (SC.Serial m a) => SC.Serial m (Out a)


-- | Encode the target structure as a vector
-- TODO: Out Double -> e.g. Out (Real Prob) or Out (Float Prob)
encode :: Out Double -> Vec8 Prob
encode Out{..} = (Vec . encode') [arcVal, hedVal, depVal]


-- | Decode the target structure from the potential vector
decode :: Vec8 Prob -> Out Double
decode vec =
  case decode' (unVec vec) of
    [arcP, hedP, depP] -> Out
      { arcVal = arcP
      , hedVal = hedP
      , depVal = depP
      }
    xs -> error $
      "Graph2.decode: unsupported list length (" ++
       show xs ++ ")"


-- | Encode a list of probabilities of length @n@ as a vector of length @2^n@.
encode'
  :: (KnownNat n)
  => [Double]
  -> R n
encode' =
  LA.vector . go
  where
    go (p:ps)
      = map (uncurry (*))
      $ cartesian [1-p, p] (go ps)
    go [] = [1]


-- | Decode the list of probabilities/potentials from the given vector.
decode'
  :: (KnownNat n)
  => R n
  -> [Double]
decode' =
  go . toList
  where
    go []  = []
    go [_] = []
    go xs =
      let (left, right) = rightInTwo xs
          -- p0 = sum left
          p1 = sum right
       in p1 : go (map (uncurry (+)) (zip left right))


-- | Cartesian product of two lists
cartesian :: [a] -> [b] -> [(a, b)]
cartesian xs ys = do
  x <- xs
  y <- ys
  return (x, y)


-- | Split a list x_1, x_2, ..., x_(2n) in two equal parts:
-- 
--   * x_1, x_2, ..., x_n, and
--   * x_(n+1), x_(n+2), ..., x_(2n)
--
rightInTwo :: [a] -> ([a], [a])
rightInTwo xs =
  List.splitAt (length xs `div` 2) xs